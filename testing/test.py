
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
from qiskit.quantum_info import Pauli, SparsePauliOp, Statevector
from qiskit.opflow import PrimitiveOp

from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister, Aer, execute, IBMQ, transpile

from qiskit.circuit import Parameter
from qiskit.transpiler.passes import RemoveBarriers
from qiskit_aer import AerSimulator

import matplotlib.pyplot as plt
from vqe_experiment import *

from qiskit.algorithms.minimum_eigensolvers import VQE 
from qiskit_aer.primitives import Estimator
from qiskit.algorithms.optimizers import ADAM, SciPyOptimizer, COBYLA, SPSA, SLSQP
from qiskit.transpiler import PassManager

from mitiq import (
    Calibrator,
    Settings,
    execute_with_mitigation,
    MeasurementResult,
)

sys.path.append("/Users/norman/Documents/GitHub/qiskit-research/")
from qiskit_research.utils.convenience import *

pi = np.pi

### Noise Model ###
import qiskit_aer.noise as noise
from qiskit.providers.aer import AerSimulator

#Noise Model: ibmq_jakarta as of 6/21
sx_err_prob = 0.0002339  # 1-qubit gate
cx_err_prob = 0.008414  # 2-qubit gate
readout_err = 0.0383  # readout error
sx_err = noise.depolarizing_error(sx_err_prob, 1)
cx_err = noise.depolarizing_error(cx_err_prob, 2)
thermalsx = noise.thermal_relaxation_error(168.7e3, 36.89e3, 50)

noise_model = noise.NoiseModel()
noise_model.add_all_qubit_quantum_error(sx_err, ['sx'])
noise_model.add_all_qubit_quantum_error(cx_err, ['cx'])
noise_model.add_all_qubit_readout_error([[1-readout_err, readout_err],[readout_err, 1-readout_err]])
noise_model.add_all_qubit_quantum_error(thermalsx, ['sx'])
#Add the noise model:
simulator = AerSimulator(noise_model=noise_model)



# result = execute(qc, backend=Aer.get_backend("qasm_simulator"), shots=4096).result()
# counts = result.get_counts()
# print(counts)


sys.path.append("/Users/norman/Documents/GitHub/BayesianOptimization/")

from bayes_opt import BayesianOptimization
from bayes_opt.logger import JSONLogger
from bayes_opt.event import Events

### Hz strength ###
h = [0.1]

e_act = []
e_cafqa = []
e_vqe = []
e_cafqa_vqe = []

reps = 2
num_param = 4*reps

save_dir = "/Users/norman/Documents/GitHub/CAFQA/testing"

for i in h:

    #Generate Hamiltonian coefficients and Paulis
    coeffs, paulis = XYmodel(i)
    n_qubits = len(paulis[0])


    logger = JSONLogger(path=save_dir+"/BOlog.log")


    # p_vec = [1,3,1,0]
    # parameters = [p*(pi/2) for p in p_vec]
    # vqe_qc = QuantumCircuit(n_qubits)
    # add_ansatz(vqe_qc, ansatz, parameters, 1)
    # print(vqe_qc)
    # vqe_qc_trans = transform_to_allowed_gates(vqe_qc)
    # print(vqe_qc_trans)



    def black_box_function(**params):

        p_vec = []

        ### Force discrete parameters ###
        for p in params.values():
            p_vec.append(int(round(p,0)))
        
        parameters = [p*(pi/2) for p in p_vec]

        vqe_qc = QuantumCircuit(n_qubits)

        ### Generates the ansatz with parameters filled ###
        add_ansatz(vqe_qc, ansatz, parameters, reps)

        ### Transforms all gates to Clifford only ###
        vqe_qc_trans = transform_to_allowed_gates(vqe_qc)
        stim_qc = qiskit_to_stim(vqe_qc_trans)
        
        sim = stim.TableauSimulator()
        sim.do_circuit(stim_qc)
        pauli_expect = [sim.peek_observable_expectation(stim.PauliString(p)) for p in paulis]

        # print(pauli_expect)
        loss = np.dot(coeffs, pauli_expect)

        # BO maximizes, so return the negative of the loss (energy)
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

    #Print the converged CAFQA energy
    print("CAFQA ground energy:"+str(-optimizer.max['target']))
    e_cafqa.append(-optimizer.max['target'])




    # Create an object to store intermediate VQE results
    from dataclasses import dataclass
    @dataclass
    class VQELog:
        values: list
        parameters: list

        def update(self, count, parameters, mean, _metadata):
            self.values.append(mean)
            self.parameters.append(parameters)
            print(f"Running circuit {count} ...", end="\r", flush=True)

    log = VQELog([], [])
    log_cafqa = VQELog([], [])

    # Define optimizer
    # opt = ADAM(maxiter=2000,amsgrad=True)
    opt = COBYLA(maxiter=1000)
    # opt = SciPyOptimizer(method='BFGS')
    # opt = SPSA(maxiter=1000)
    # opt = SLSQP(maxiter=1000)

    # Define hamiltonian
    Ham = PrimitiveOp(SparsePauliOp(paulis,coeffs = coeffs))
    
    # backend = Aer.get_backend('statevector_simulator')
    # backend = sim_vigo
    backend = simulator


    ### Save the optimal CAFQA parameterss
    p_vec = []
    for p in optimizer.max['params'].values():
        p_vec.append(int(round(p,0)))
    
    parameters = [p*(pi/2) for p in p_vec]

    # Generate the ansatz
    qc,n = ansatz(4,2)

    num_twirl=10
    # print(qc)
    twirlqc = add_pauli_twirls(qc,num_twirl)
    # ddtwirlqc = add_dynamical_decoupling(twirlqc[0],backend,'XY4',add_pulse_cals=True)


    estimator = Estimator(backend_options={"noise_model":noise_model})

    # Set up VQE and run on backend
    vqe_only = VQE(estimator=estimator,ansatz=qc, optimizer=opt, callback=log.update)
    result = vqe_only.compute_minimum_eigenvalue(operator=Ham)
    vqe_cafqa = VQE(estimator=estimator,ansatz=qc, optimizer=opt, initial_point=parameters, callback=log_cafqa.update)
    result_w_cafqa = vqe_cafqa.compute_minimum_eigenvalue(operator=Ham)

    plt.plot(log.values, label="VQE")
    plt.plot(log_cafqa.values, label="CAFQA + VQE")
    plt.plot([get_ref_energy(coeffs, paulis) for i in range(len(log.values))],"-k",label="Actual")

    # print the results
    print("VQE energy: "+str(result.eigenvalue.real))
    e_vqe.append(result.eigenvalue.real)
    print("CAFQA + VQE energy: "+str(result_w_cafqa.eigenvalue.real))
    e_cafqa_vqe.append(result_w_cafqa.eigenvalue.real)


    print("Actual energy:"+str(get_ref_energy(coeffs, paulis)))
    e_act.append(get_ref_energy(coeffs, paulis))

# plt.plot(h,e_act,"--k",label="Actual")
# plt.plot(h,e_cafqa,"o",label="CAFQA")
# plt.plot(h,e_vqe,"o",label="VQE")
# plt.plot(h,e_cafqa_vqe,"o",label="CAFQA + VQE")
plt.xlabel('Iter')
plt.ylabel('Energy')
plt.legend()
plt.title("4-site XXZ Model: XX + YY + Hz*ZZ (Hz = "+str(h[0])+")")
plt.savefig("mygraph.png")

# # %%
# round(0.50001,0)
# # %%

# %%
