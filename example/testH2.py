import numpy as np

from qiskit_nature.units import DistanceUnit
from qiskit_nature.second_q.circuit.library import HartreeFock
from qiskit_nature.second_q.transformers import ActiveSpaceTransformer
from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit_nature.second_q.mappers import ParityMapper, QubitConverter
from qiskit_nature.second_q.hamiltonians import FermiHubbardModel
from qiskit_nature.second_q.hamiltonians.lattices import LineLattice, BoundaryCondition
from qiskit import QuantumCircuit
from qiskit.extensions import HamiltonianGate

from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister, Aer, execute, IBMQ, transpile
from qiskit.circuit import Parameter
from qiskit.transpiler.passes import RemoveBarriers
from qiskit.providers.aer import AerSimulator

import hypermapper
import json
import sys
import matplotlib.pyplot as plt
from qiskit.providers.fake_provider import FakeMumbai
from vqe_experiment import *

from bayes_opt import BayesianOptimization

pi = np.pi

bond_length = 0.757
# H2 molecule string
atom_string = f"H 0.0 0.0 0.0; H 0.0 {bond_length} 0.0"
num_orbitals = 2
coeffs, paulis, HF_bitstring = molecule(atom_string, num_orbitals)
n_qubits = len(paulis[0])

print(coeffs,paulis)

save_dir = "./"
result_file = "result.txt"
budget = 500
vqe_kwargs = {
    "ansatz_reps": 2,
    "init_last": False,
    "HF_bitstring": HF_bitstring
}


loss_file = "cafqa_loss.txt"
params_file = "cafqa_params.txt"

x_0 = 0
x_1 = 0
x_2 = 0
x_3 = 0
x_4 = 0
x_5 = 0
x_6 = 0
x_7 = 0
x_8 = 0
x_9 = 0
x_10 = 0
x_11 = 0


parameters = [x_0*(np.pi/2),x_1*(np.pi/2),x_2*(np.pi/2),x_3*(np.pi/2),x_4*(np.pi/2),x_5*(np.pi/2),x_6*(np.pi/2),x_7*(np.pi/2),x_8*(np.pi/2),x_9*(np.pi/2),x_10*(np.pi/2),x_11*(np.pi/2)]
# vqe_qc = QuantumCircuit(n_qubits)
# circ,p = efficientsu2_full(n_qubits,2)
# # print(circ)
# add_ansatz(vqe_qc, efficientsu2_full, parameters, vqe_kwargs['ansatz_reps'])
# vqe_qc_trans = transform_to_allowed_gates(vqe_qc)
# # print(vqe_qc_trans)
# # print([vqe_qc_trans[i].operation.name.upper() for i in range(len(vqe_qc_trans))])

# allowed_gates = ["X", "Y", "Z", "H", "CX", "S", "S_DAG", "SQRT_X", "SQRT_X_DAG"]
# stim_circ = stim.Circuit()
# # make sure right number of qubits in stim circ
# for i in range(vqe_qc_trans.num_qubits):
#     stim_circ.append("I", [i])
# for instruction in vqe_qc_trans:
#     gate_lbl = instruction.operation.name.upper()
#     if gate_lbl == "BARRIER":
#         continue
#     elif gate_lbl == "SDG":
#         gate_lbl = "S_DAG"
#     elif gate_lbl == "SX":
#         gate_lbl = "SQRT_X"
#     elif gate_lbl == "SXDG":
#         gate_lbl = "SQRT_X_DAG"
#     assert gate_lbl in allowed_gates, f"Invalid gate {gate_lbl}."
#     qubit_idc = [qb.index for qb in instruction.qubits]
#     stim_circ.append(gate_lbl, qubit_idc)
# # print(stim_circ)

# sim = stim.TableauSimulator()
# sim.do_circuit(stim_circ)
# pauli_expect = [sim.peek_observable_expectation(stim.PauliString(p)) for p in paulis]
# # print(coeffs,pauli_expect)
# loss = np.dot(coeffs, pauli_expect)
# # print(loss)

def black_box_function(**params):

    start = timer()

    p_vec = []

    for p in params.values():
        p_vec.append(int(round(p,0)))
    # print(p_vec)
    parameters = [p*(pi/2) for p in p_vec]


    vqe_qc = QuantumCircuit(n_qubits)
    add_ansatz(vqe_qc, efficientsu2_full, parameters, vqe_kwargs['ansatz_reps'])
    vqe_qc_trans = transform_to_allowed_gates(vqe_qc)
    stim_qc = qiskit_to_stim(vqe_qc_trans)
    
    sim = stim.TableauSimulator()
    sim.do_circuit(stim_qc)
    pauli_expect = [sim.peek_observable_expectation(stim.PauliString(p)) for p in paulis]
    loss = np.dot(coeffs, pauli_expect)
    end = timer()
    # print(f'Loss computed by CAFQA VQE is {loss}, in {end - start} s.')
    loss_filename = save_dir + "/" + loss_file
    params_filename = save_dir + "/" + params_file
    if loss_filename is not None:
        with open(loss_filename, 'a') as file:
            writer = csv.writer(file)
            writer.writerow([loss])
    
    if params_filename is not None and parameters is not None:
        with open(params_filename, 'a') as file:
            writer = csv.writer(file)
            writer.writerow(parameters)
    return -loss

# Bounded region of parameter space
pbounds = {'x_'+str(i): (0,3) for i in range(len(parameters))}

optimizer = BayesianOptimization(
    f = black_box_function,
    pbounds=pbounds,
    verbose=2, # verbose = 1 prints only when a maximum is observed, verbose = 0 is silent
    random_state=0,
)


optimizer.maximize(
    init_points=len(parameters),
    n_iter=3,
)

param_vec = [optimizer.max['params']['x_'+str(i)] for i in range(len(optimizer.max['params']))]
vqe_qc = QuantumCircuit(n_qubits)
qc,p = efficientsu2_full(n_qubits,2)
# print(qc.parameters)
# print(optimizer.max['params'])
qc.assign_parameters(parameters=param_vec, inplace=True)
qc.compose(qc, inplace=True)
# print(qc)

print("Optimizer result: "+str(-optimizer.max['target']))
print("Reference value: "+str(get_ref_energy(coeffs, paulis)))