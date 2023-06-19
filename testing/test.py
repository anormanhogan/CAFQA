
#%%
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

pi = np.pi

# result = execute(qc, backend=Aer.get_backend("qasm_simulator"), shots=4096).result()
# counts = result.get_counts()
# print(counts)

from bayes_opt import BayesianOptimization
from bayes_opt.logger import JSONLogger
from bayes_opt.event import Events
# x_0 = 1
# x_1 = 1
# x_2 = 1
# x_3 = 1
# x_4 = 1
# x_5 = 1
# x_6 = 1
# x_7 = 1

# # parameters = [x_0*(np.pi/2),x_1*(np.pi/2),x_2*(np.pi/2),x_3*(np.pi/2),x_4*(np.pi/2),x_5*(np.pi/2),x_6*(np.pi/2),x_7*(np.pi/2)]
# circuit = QuantumCircuit(2,2)
# n_qubits = circuit.num_qubits
# circuit, _ = ansatz(n_qubits, 2)
# # print(circuit)

# # circuit.assign_parameters(parameters=parameters, inplace=True)
# # circuit.compose(circuit, inplace=True)
# print(circuit)

h = [0.5*i for i in range(0,5)]

e_act = []
e_cafqa = []

for i in h:
    coeffs, paulis = XYmodel(i)
    n_qubits = len(paulis[0])

    save_dir = "/Users/norman/Documents/GitHub/CAFQA/testing"
    result_file = "result.txt"
    budget = 500
    vqe_kwargs = {
        "ansatz_reps": 2,
        "init_last": False,
        "HF_bitstring": '00'
    }

    logger = JSONLogger(path=save_dir+"/BOlog.log")

    # run CAFQA
    cafqa_guess = [] # will start from all 0 parameters
    loss_file = "cafqa_loss.txt"
    params_file = "cafqa_params.txt"
    loss_filename = save_dir + "/" + loss_file
    params_filename = save_dir + "/" + params_file
    num_param = vqe_kwargs["ansatz_reps"]*(3*(n_qubits-1) + 1)

    # x_0 = 3
    # x_1 = 1
    # x_2 = 2
    # x_3 = 3
    # x_4 = 3
    # x_5 = 1
    # x_6 = 2
    # x_7 = 3
    # x_8 = 3
    # x_9 = 1
    # x_10 = 2
    # x_11 = 3


    # parameters = [x_0*(np.pi/2),x_1*(np.pi/2),x_2*(np.pi/2),x_3*(np.pi/2),x_4*(np.pi/2),x_5*(np.pi/2),x_6*(np.pi/2),x_7*(np.pi/2),x_8*(np.pi/2),x_9*(np.pi/2),x_10*(np.pi/2),x_11*(np.pi/2)]
    # qc = QuantumCircuit(n_qubits,n_qubits)
    # params_per_rep = 3*(n_qubits-1)+1

    # for i in range (vqe_kwargs["ansatz_reps"]):
    #     if i == 0:
    #         paramvec = np.array([Parameter("x_"+str(p)) for p in range(params_per_rep)])
    #     else:
    #         newparams = np.array([Parameter("x_"+str(p + i*params_per_rep)) for p in range(params_per_rep)])
    #         paramvec = np.append(paramvec,newparams)

    #     count = 0

    #     for qubit in range(0,n_qubits-1,2):
    #         qc.rxx(paramvec[i*params_per_rep + count],qubit,qubit+1)
    #         count+=1
    #     for qubit in range(1,n_qubits-1,2):
    #         qc.rxx(paramvec[i*params_per_rep + count],qubit,qubit+1)
    #         count+=1

    #     for qubit in range(0,n_qubits-1,2):
    #         qc.ryy(paramvec[i*params_per_rep + count],qubit,qubit+1)
    #         count+=1
    #     for qubit in range(1,n_qubits-1,2):
    #         qc.ryy(paramvec[i*params_per_rep + count],qubit,qubit+1)
    #         count+=1

    #     for qubit in range(n_qubits-1):
    #         if qubit == 0:
    #             qc.h(qubit)
    #             qc.rz(paramvec[i*params_per_rep + count],qubit)
    #             qc.h(qubit)
    #             count+=1
    #             qc.h(qubit+1)
    #             qc.rz(paramvec[i*params_per_rep + count],qubit+1)
    #             qc.h(qubit+1)
    #             count+=1
    #         else:
    #             qc.h(qubit+1)
    #             qc.rz(paramvec[i*params_per_rep + count],qubit+1)
    #             qc.h(qubit+1)
    #             count+=1

    # # qc,n_p = ansatz(n_qubits,vqe_kwargs["ansatz_reps"])
    # print(qc)
    # vqe_qc = QuantumCircuit(n_qubits)
    # add_ansatz(vqe_qc, ansatz, parameters, vqe_kwargs['ansatz_reps'])
    # print(vqe_qc)
    # vqe_qc_trans = transform_to_allowed_gates(vqe_qc)
    # print(vqe_qc_trans)
    # print([vqe_qc_trans[i].operation.name.upper() for i in range(len(vqe_qc_trans))])

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
    # print(stim_circ)

    # sim = stim.TableauSimulator()
    # sim.do_circuit(stim_circ)
    # pauli_expect = [sim.peek_observable_expectation(stim.PauliString(p)) for p in paulis]
    # print(coeffs,pauli_expect)
    # loss = np.dot(coeffs, pauli_expect)
    # print(loss)

    def black_box_function(**params):

        start = timer()

        p_vec = []

        for p in params.values():
            p_vec.append(int(round(p,0)))
        # print(p_vec)
        parameters = [p*(pi/2) for p in p_vec]

        vqe_qc = QuantumCircuit(n_qubits)
        add_ansatz(vqe_qc, ansatz, parameters, vqe_kwargs['ansatz_reps'])
        vqe_qc_trans = transform_to_allowed_gates(vqe_qc)
        stim_qc = qiskit_to_stim(vqe_qc_trans)
        
        sim = stim.TableauSimulator()
        sim.do_circuit(stim_qc)
        pauli_expect = [sim.peek_observable_expectation(stim.PauliString(p)) for p in paulis]
        # print(pauli_expect)
        loss = np.dot(coeffs, pauli_expect)
        end = timer()
        # print(f'Loss computed by CAFQA VQE is {loss}, in {end - start} s.')
        
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
    pbounds = {'x_'+str(i): (0,3) for i in range(num_param)}

    optimizer = BayesianOptimization(
        f = black_box_function,
        pbounds=pbounds,
        verbose=2, # verbose = 1 prints only when a maximum is observed, verbose = 0 is silent
        random_state=0,
    )

    # optimizer.subscribe(Events.OPTIMIZATION_STEP, logger)

    optimizer.maximize(
        init_points=num_param,
        n_iter=10,
    )

    param_vec = [optimizer.max['params']['x_'+str(i)] for i in range(len(optimizer.max['params']))]
    vqe_qc = QuantumCircuit(n_qubits)
    qc,p = ansatz(n_qubits,vqe_kwargs['ansatz_reps'])
    # print(qc.parameters)
    print("CAFQA ground energy:"+str(-optimizer.max['target']))
    e_cafqa.append(-optimizer.max['target'])
    qc.assign_parameters(parameters=param_vec, inplace=True)
    qc.compose(qc, inplace=True)
    # print(qc)

    print("Actual energy:"+str(get_ref_energy(coeffs, paulis)))
    e_act.append(get_ref_energy(coeffs, paulis))

plt.plot(h,e_act,label="Actual")
plt.plot(h,e_cafqa,label="CAFQA")
plt.xlabel('Bx')
plt.ylabel('Energy')
plt.legend()
plt.savefig("mygraph.png")

# %%
round(0.50001,0)
# %%
