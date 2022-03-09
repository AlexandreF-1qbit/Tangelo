# Copyright 2021 Good Chemistry Company.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Docstring.
"""

import numpy as np

from tangelo.toolboxes.ansatz_generator.ansatz import Ansatz
from tangelo.toolboxes.qubit_mappings.mapping_transform import get_qubit_number
from tangelo.toolboxes.qubit_mappings.statevector_mapping import get_reference_circuit
from tangelo.linq import Circuit, Gate


class SymmetricHEA(Ansatz):
    def __init__(self, molecule=None, mapping="jw", up_then_down=False,
                n_layers=2, sym="both", n_qubits=None, n_electrons=None,
                reference_state="HF"):

        if not (bool(molecule) ^ (bool(n_qubits) and bool(n_electrons))):
            raise ValueError(f"A molecule OR qubit + electrons number must be provided when instantiating the HEA.")

        if n_qubits:
            self.n_qubits = n_qubits
            self.n_electrons = n_electrons
        else:
            self.n_qubits = get_qubit_number(mapping, molecule.n_active_sos)
            self.n_electrons = molecule.n_active_electrons

        self.qubit_mapping = mapping
        self.up_then_down = up_then_down
        self.n_layers = n_layers
        self.reference_state = reference_state
        self.sym = sym

        # Number of qubits must be even.
        assert self.n_qubits % 2 == 0

        if self.sym == "both":
            self.n_var_params = (self.n_qubits - 1) * self.n_layers
        elif self.sym == "sz":
            self.n_var_params = (2*self.n_qubits - 1) * self.n_layers
        else:
            raise ValueError("Unsuported option.")

        # Supported reference state initialization
        self.supported_reference_state = {"HF"}
        # Supported var param initialization
        self.supported_initial_var_params = {"ones", "random", "zeros"}

        # Default initial parameters for initialization
        self.var_params_default = "random"

        self.var_params = None
        self.circuit = None

    def set_var_params(self, var_params=None):
        if var_params is None:
            var_params = self.var_params_default

        if isinstance(var_params, str):
            if (var_params not in self.supported_initial_var_params):
                raise ValueError(f"Supported keywords for initializing variational parameters: {self.supported_initial_var_params}")
            else:
                if var_params == "ones":
                    initial_var_params = np.ones((self.n_var_params,), dtype=float)
                elif var_params == "random":
                    initial_var_params = 2 * np.pi * np.random.random((self.n_var_params,))
                elif var_params == "zeros":
                    initial_var_params = np.zeros((self.n_var_params,), dtype=float)
        else:
            initial_var_params = np.array(var_params)
            if initial_var_params.size != self.n_var_params:
                raise ValueError(f"Expected {self.n_var_params} variational parameters but "
                                 f"received {initial_var_params.size}.")
        self.var_params = initial_var_params
        return initial_var_params

    def prepare_reference_state(self):
        if self.reference_state not in self.supported_reference_state:
            raise ValueError(f"{self.reference_state} not in supported reference state methods of:{self.supported_reference_state}")

        if self.reference_state == "HF":
            return get_reference_circuit(n_spinorbitals=self.n_qubits,
                                         n_electrons=self.n_electrons,
                                         mapping=self.qubit_mapping,
                                         up_then_down=self.up_then_down)

    def build_circuit(self, var_params=None):
        self.var_params = self.set_var_params(var_params)

        reference_state_circuit = self.prepare_reference_state()

        symhea_circuit = self.construct_symhea_circuit()

        if reference_state_circuit.size != 0:
            self.circuit = reference_state_circuit + symhea_circuit
        else:
            self.circuit = symhea_circuit

        self.update_var_params(self.var_params)
        return self.circuit

    def update_var_params(self, var_params):
        self.set_var_params(var_params)
        var_params = self.var_params

        var_params = np.array(np.split(var_params, self.n_layers))
        if self.sym == "sz":
            params_phase = var_params[:,-self.n_qubits:]
            params_entanglers = var_params[:,:-self.n_qubits]
        else:
            params_entanglers = var_params

        n_params_entangler_per_layer = 3*params_entanglers.shape[1]
        theta = 2*params_entanglers.flatten()[:, np.newaxis] - np.pi/2
        p = np.hstack((theta, -theta, theta))
        params_entanglers = np.reshape(p, (self.n_layers, n_params_entangler_per_layer))

        if self.sym == "sz":
            all_params = np.concatenate((params_entanglers, params_phase), axis=1)
        else:
            all_params = params_entanglers

        all_params = all_params.flatten()

        for param_index, var_gate in enumerate(self.circuit._variational_gates):
            var_gate.parameter = all_params[param_index]

    def construct_symhea_circuit(self):
        symhea_gates = list()

        for _ in range(self.n_layers):
            for i_qubit in range(0, self.n_qubits, 2):
                symhea_gates += self.weird_N(0., i_qubit, i_qubit + 1)

            for i_qubit in range(1, self.n_qubits-1, 2):
                symhea_gates += self.weird_N(0., i_qubit, i_qubit + 1)

            if self.sym == "sz":
                symhea_gates += [Gate("PHASE", target=i, is_variational=True) for i in range(self.n_qubits)]

        return Circuit(symhea_gates)

    @staticmethod
    def weird_N(theta, qubit_i, qubit_j):
        demi_pi = np.pi/2

        gates = [Gate("RZ", target=qubit_j, parameter=demi_pi)]
        gates += [Gate("CNOT", target=qubit_i, control=qubit_j)]
        gates += [Gate("RZ", target=qubit_i, parameter=2*theta-demi_pi, is_variational=True)]
        gates += [Gate("RY", target=qubit_j, parameter=demi_pi-2*theta, is_variational=True)]
        gates += [Gate("CNOT", target=qubit_j, control=qubit_i)]
        gates += [Gate("RY", target=qubit_j, parameter=2*theta-demi_pi, is_variational=True)]
        gates += [Gate("CNOT", target=qubit_i, control=qubit_j)]
        gates += [Gate("RZ", target=qubit_i, parameter=-demi_pi)]

        return gates
