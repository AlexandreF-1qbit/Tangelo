# Copyright 2023 Good Chemistry Company.
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

"""This module defines functions to get the particle-hole Hamiltonian of a
molecular system.
"""

import numpy as np

from tangelo.toolboxes.operators import FermionOperator


EQ_TOLERANCE = 1e-8

# Source: https://github.com/goodchemistryco/Feynman/blob/6d2552d8d775426a5348f451ee5cf7dac5001566/Lee_code/p_h_hamiltonian/_molecular_data.py#L940
def get_normal_ordered_particle_hole_hamiltonian(mol):

    constant, one_body_integrals, two_body_integrals = mol.get_active_space_integrals()
    n_qubits = 2 * one_body_integrals.shape[0]

    # Initialize Hamiltonian coefficients.
    one_body_coefficients = np.zeros((n_qubits, n_qubits))
    two_body_coefficients = np.zeros((n_qubits, n_qubits,
                                      n_qubits, n_qubits))
    # Loop through integrals.
    for p in range(n_qubits // 2):
        for q in range(n_qubits // 2):

            # Populate 1-body coefficients. Require p and q have same spin.
            one_body_coefficients[2 * p, 2 * q] = one_body_integrals[p, q]
            one_body_coefficients[2 * p + 1, 2 * q + 1] = one_body_integrals[p, q]

            # Continue looping to prepare 2-body coefficients (we use the usual convention).
            for r in range(n_qubits // 2):
                for s in range(n_qubits // 2):

                    # Require p,r and q,s to have same spin. Handle mixed
                    # spins.
                    two_body_coefficients[2 * p, 2 * q + 1, 2 * r,
                                            2 * s + 1] = (
                            two_body_integrals[p, q, s, r])
                    two_body_coefficients[2 * p + 1, 2 * q, 2 * r + 1,
                                            2 * s] = (
                            two_body_integrals[p, q, s, r])

                    # Avoid having two electrons in same orbital. Handle
                    # same spins.
                    if p != q and r != s:
                        two_body_coefficients[2 * p, 2 * q, 2 * r,
                                                2 * s] = (
                                two_body_integrals[p, q, s, r])
                        two_body_coefficients[2 * p + 1, 2 * q + 1,
                                                2 * r + 1, 2 * s + 1] = (
                                two_body_integrals[p, q, s, r])

    # Truncate.
    one_body_coefficients[np.absolute(one_body_coefficients) < EQ_TOLERANCE] = 0.
    two_body_coefficients[np.absolute(two_body_coefficients) < EQ_TOLERANCE] = 0.

    # Define spinorbital ranges
    i0 = 0
    if mol.active_occupied:
        i0 = 2*len(mol.frozen_occupied)

    n_occ = mol.n_active_electrons
    n_virt = n_qubits - n_occ

    hamiltonian = FermionOperator()

    # One particle part (assume a canonical HF reference for now)
    # --------------------------------------------------------------

    # p-p sector
    for a in range(n_virt):
        a_1 = a + n_occ

        for b in range(n_virt):
            b_1 = b + n_occ

            # Compute the element of the one-particle part of the Fock matrix
            Fvv = one_body_coefficients[a_1, b_1]

            for i in range(n_occ):
                i_1 = i + i0

                # Add the two-particle terms to the Fock matrix
                Fvv += two_body_coefficients[a_1, i_1, b_1, i_1] - two_body_coefficients[a_1, i_1, i_1, b_1]

            # Add the contribution to the Hamiltonian
            hamiltonian += FermionOperator(((a_1, 1), (b_1, 0)), Fvv)

    # h-h sector
    for i in range(n_occ):
        i_1 = i + i0

        for j in range(n_occ):
            j_1 = j + i0

            # Compute the element of the one-particle part of the Fock matrix
            Foo = one_body_coefficients[i_1, j_1]

            for k in range(n_occ):
                k_1 = k + i0

                # Add the two-particle terms to the Fock matrix
                Foo += two_body_coefficients[i_1, k_1, j_1, k_1] - two_body_coefficients[i_1, k_1, k_1, j_1]

            hamiltonian += FermionOperator(((j_1, 0), (i_1, 1)), -Foo)

    # Two particle part
    # TO DO: Loop fusion
    # --------------------------------------------------------------

    # p-p-p-p sector
    for a in range(n_virt):
        a_1 = a + n_occ

        for b in range(n_virt):
            b_1 = b + n_occ

            for c in range(n_virt):
                c_1 = c + n_occ

                for d in range(n_virt):
                    d_1 = d + n_occ

                    # Compute antisymmetric two-electron integrals
                    integ = two_body_coefficients[a_1, b_1, c_1, d_1] \
                                - two_body_coefficients[a_1, b_1, d_1, c_1]

                    # Add contribution to the Hamiltonian
                    hamiltonian += FermionOperator(((a_1, 1), (b_1, 1), (d_1, 0), (c_1, 0)), 0.25*integ)

    # h-h-h-h sector
    for i in range(n_occ):
        i_1 = i + i0

        for j in range(n_occ):
            j_1 = j + i0

            for k in range(n_occ):
                k_1 = k + i0

                for l in range(n_occ):
                    l_1 = l + i0

                    # Compute antisymmetric two-electron integrals
                    integ = two_body_coefficients[i_1, j_1, k_1, l_1] \
                            - two_body_coefficients[i_1, j_1, l_1, k_1]

                    # Add contribution to the Hamiltonian
                    hamiltonian += FermionOperator(((l_1, 0), (k_1, 0), (i_1, 1), (j_1, 1)), 0.25 * integ)

    # p-h-p-h, p-p-h-h and h-h-p-p sectors
    for i in range(n_occ):
        i_1 = i + i0

        for j in range(n_occ):
            j_1 = j + i0

            for a in range(n_virt):
                a_1 = a + n_occ

                for b in range(n_virt):
                    b_1 = b + n_occ

                    # --------
                    # p-h-p-h:
                    # ---------

                    # Compute antisymmetric two-electron integrals
                    integ = two_body_coefficients[a_1, i_1, b_1, j_1] \
                            - two_body_coefficients[a_1, i_1, j_1, b_1]

                    # Add contribution to the Hamiltonian
                    hamiltonian += FermionOperator(((a_1, 1), (j_1, 0), (b_1, 0), (i_1, 1)), integ)

                    # --------
                    # p-p-h-h:
                    # ---------

                    # Compute antisymmetric two-electron integrals
                    integ = two_body_coefficients[a_1, b_1, i_1, j_1] \
                            - two_body_coefficients[a_1, b_1, j_1, i_1]

                    # Add contribution to the Hamiltonian
                    hamiltonian += FermionOperator(((a_1, 1), (b_1, 1), (j_1, 0), (i_1, 0)), 0.25*integ)

                    # --------
                    # h-h-p-p:
                    # ---------

                    # Compute antisymmetric two-electron integrals
                    integ = two_body_coefficients[i_1, j_1, a_1, b_1] \
                            - two_body_coefficients[i_1, j_1, b_1, a_1]

                    # Add contribution to the Hamiltonian
                    hamiltonian += FermionOperator(((i_1, 1), (j_1, 1), (b_1, 0), (a_1, 0)), 0.25*integ)

    # p-p-p-h and p-h-p-p sectors
    for i in range(n_occ):
        i_1 = i + i0

        for a in range(n_virt):
            a_1 = a + n_occ

            for b in range(n_virt):
                b_1 = b + n_occ

                for c in range(n_virt):
                    c_1 = c + n_occ

                    # --------
                    # p-p-p-h:
                    # ---------

                    # Compute antisymmetric two-electron integrals
                    integ = two_body_coefficients[a_1, b_1, c_1, i_1] \
                            - two_body_coefficients[a_1, b_1, i_1, c_1]

                    # Add contribution to the Hamiltonian
                    hamiltonian += FermionOperator(((a_1, 1), (b_1, 1), (i_1, 0), (c_1, 0)), 0.5 * integ)

                    # --------
                    # p-h-p-p:
                    # ---------

                    # Compute antisymmetric two-electron integrals
                    integ = two_body_coefficients[a_1, i_1, b_1, c_1] \
                            - two_body_coefficients[a_1, i_1, c_1, b_1]

                    # Add contribution to the Hamiltonian
                    hamiltonian += FermionOperator(((a_1, 1), (i_1, 1), (c_1, 0), (b_1, 0)), 0.5 * integ)

    # p-h-h-h and h-h-h-p  sectors
    for i in range(n_occ):
        i_1 = i + i0

        for j in range(n_occ):
            j_1 = j + i0

            for k in range(n_occ):
                k_1 = k + i0

                for a in range(n_virt):
                    a_1 = a + n_occ

                    # --------
                    # p-h-h-h:
                    # ---------

                    # Compute antisymmetric two-electron integrals
                    integ = two_body_coefficients[a_1, i_1, j_1, k_1] \
                            - two_body_coefficients[a_1, i_1, k_1, j_1]

                    # Add contribution to the Hamiltonian
                    hamiltonian += FermionOperator(((a_1, 1), (k_1, 0), (j_1, 0), (i_1, 1)), 0.5 * integ)

                    # --------
                    # h-h-p-h:
                    # ---------

                    # Compute antisymmetric two-electron integrals
                    integ = two_body_coefficients[i_1, j_1, a_1, k_1] \
                            - two_body_coefficients[i_1, j_1, k_1, a_1]

                    # Add contribution to the Hamiltonian
                    hamiltonian += FermionOperator(((k_1, 0), (i_1, 1), (j_1, 1), (a_1, 0)), 0.5 * integ)

    return hamiltonian + mol.mf_energy
