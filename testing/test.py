
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
from qiskit.algorithms.optimizers import ADAM, SciPyOptimizer, COBYLA, SPSA, SLSQP, AQGD, GradientDescent, NFT
from qiskit.transpiler import PassManager

from qiskit_ibm_provider import IBMProvider

from mitiq import (
    Calibrator,
    Settings,
    execute_with_mitigation,
    MeasurementResult,
)

sys.path.append("/Users/norman/Documents/GitHub/qiskit-research/")
from qiskit_research.utils.convenience import *

pi = np.pi
token = '73a23d4c073e79f7b99b2ad9fc68dca92d54355a2433d444fe70fda0c12ac9ad49fa3ab738591ce75235b07be15bea552e1e6d14e99318f75eb33c1b121f30b5'

### Noise Model ###
import qiskit_aer.noise as noise
from qiskit.providers.aer import AerSimulator

def make_noise_model():
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
    return noise_model
#Add the noise model:
noise_model = make_noise_model()
simulator = AerSimulator(method='statevector',noise_model=noise_model)



# result = execute(qc, backend=Aer.get_backend("qasm_simulator"), shots=4096).result()
# counts = result.get_counts()
# print(counts)


sys.path.append("/Users/norman/Documents/GitHub/BayesianOptimization/")

from bayes_opt import BayesianOptimization
from bayes_opt.logger import JSONLogger
from bayes_opt.event import Events

### Hz strength ###
# h = [np.round(0.15*i,2) for i in range(10)]
h = [0.1 for _ in range(30)]

e_act = []
e_cafqa = []
e_vqe = []
e_cafqa_vqe = []

vqes = []
cafqa_vqes = []

reps = 2
num_param = 4*reps

save_dir = "/Users/norman/Documents/GitHub/CAFQA/testing"
loss_file = "vqe_cafqa_loss.txt"
params_file = "vqe_cafqa_params.txt"

vqe_kwargs = {
    "ansatz_reps": reps,
    "init_last": False,
    "HF_bitstring": "1010"
}

def get_PT_obs(twirled_circuits,observable):
    shots = 10**4
    expectation_value = 0
    count = 1
    for circ in twirled_circuits:
        circ.measure_all()
        print(f"Running circuit {count} of "+str(len(twirled_circuits)), end="\r", flush=True)
        result = backend.run(circ, shots=shots).result()
        count+=1
        counts = result.get_counts(circ)
        counts = { k.replace(" 0000",""):v for k, v in counts.items()}
        print(counts)
        keys = []
        for i in range(2**4):
            b = bin(i)[2:]
            l = len(b)
            keys.append(str(0) * (4 - l) + b)
        # print(keys)
        counts_sorted = []
        for i in keys:
            try:
                counts_sorted.append(counts[i]/shots)
            except:
                counts_sorted.append(0)
        # print(noisy_counts_sorted)
        state = Statevector(counts_sorted)
        # state = Statevector(np.linalg.eigh(observable)[1][:,0])
        # print(state)
        expectation_value += state.expectation_value(observable)
    expectation_value /= len(twirled_circuits)

    return expectation_value


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
def XYmodel(h):
    """
    Compute Hamiltonian for molecule in qubit encoding using Qiskit Nature.
    atom_string (String): string to describe molecule, passed to PySCFDriver.
    new_num_orbitals (Int): Number of orbitals in active space (if None, use default result from PySCFDriver).
    kwargs (Dict): All the arguments that need to be passed on to the next function calls.

    Returns:
    (Iterable[Float], Iterable[String], String) (Pauli coefficients, Pauli strings, Hartree-Fock bitstring)
    """
    # converter = QubitConverter(ParityMapper(), two_qubit_reduction=True)
    # driver = PySCFDriver(
    #     atom=atom_string,
    #     basis="sto3g",
    #     charge=0,
    #     spin=0,
    #     unit=DistanceUnit.ANGSTROM
    # )
    # problem = driver.run()
    # if new_num_orbitals is not None:
    #     num_electrons = (problem.num_alpha, problem.num_beta)
    #     transformer = ActiveSpaceTransformer(num_electrons, new_num_orbitals)
    #     problem = transformer.transform(problem)
    # ferOp = problem.hamiltonian.second_q_op()
    # qubitOp = converter.convert(ferOp, problem.num_particles)
    # initial_state = HartreeFock(
    #     problem.num_spatial_orbitals,
    #     problem.num_particles,
    #     converter
    # )
    # bitstring = "".join(["1" if bit else "0" for bit in initial_state._bitstr])
    # # need to reverse order bc of qiskit endianness
    # paulis = [x[::-1] for x in qubitOp.primitive.paulis.to_labels()]
    # # add the shift as extra I pauli
    # paulis.append("I"*len(paulis[0]))
    # paulis = np.array(paulis)
    # coeffs = list(qubitOp.primitive.coeffs)
    # # add the shift (nuclear repulsion)
    # coeffs.append(problem.nuclear_repulsion_energy)
    # coeffs = np.array(coeffs).real

    coeffs = np.array([h,h,1,1,1,1,h,h,1,1,1,1])
    paulis = np.array(['ZIIZ','IZZI','YIIY','IYYI','XIIX','IXXI','ZZII','IIZZ','XXII','IIXX','YYII','IIYY'])

    return coeffs, paulis

###### Cycle through adjustable parameter Hz
for i in h:

    print('##############################################')
    print('Beginning procedure for Hz = '+str(i)+' ...')
    print('##############################################')

    ####### Generate Hamiltonian coefficients and Paulis
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

    ######## Bounded region of parameter space 
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

    ####### Print the converged CAFQA energy
    print("CAFQA ground energy: "+str(-optimizer.max['target']))
    e_cafqa.append(-optimizer.max['target'])




    ######## Create an object to store intermediate VQE results
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

    ######## Define Hamiltonian #######
    Ham = SparsePauliOp(paulis,coeffs = coeffs)
    # Ham = SparsePauliOp(paulis,coeffs = coeffs).to_matrix()
    print("Actual ground energy: "+str(np.linalg.eigh(Ham)[0][0]))
    # print("Actual GS: "+str(np.linalg.eigh(Ham)[1][:,0]))

    ########## Save the optimal CAFQA parameters #########
    cafqa_parameters = [int(round(p,0))*(pi/2) for p in optimizer.max['params'].values()]
    # print(parameters)

    ########## Generate the ansatz #########
    # qc = QuantumCircuit(4,4)
    # add_ansatz(qc,ansatz,parameters,2)
    # print(qc)
    qc,n = ansatz(n_qubits,reps)

    ########## Generate twirled circuits ##########
    # num_twirl=3
    # print(qc)
    # twirlqc = add_pauli_twirls(qc,num_twirl)
    # ddtwirlqc = add_dynamical_decoupling(twirlqc[0],backend,'XY4',add_pulse_cals=True)

    ######## GETTING BACKEND ########
    IBMProvider.delete_account()
    IBMProvider.save_account(instance="ibm-q/open/main",token=token, overwrite=True)
    provider = IBMProvider()
    backend = provider.get_backend('simulator_statevector',noise_model=None)
    # backend = sim_vigo
    # backend = simulator

    ########## Initialize Estimator ########
    #PRIMITIVES
    from qiskit.primitives import Estimator
    shots = 20000
    estimator = Estimator()
    estimator.set_options(shots=shots)
    # estimator.set_options(shots=shots, noise_model=noise_model)

    #RUNTIME
    # from qiskit_ibm_runtime import Estimator, Session, QiskitRuntimeService, Options
    # service = QiskitRuntimeService(channel='ibm_quantum',token=token)
    # backend = service.get_backend('simulator_statevector')
    # options = Options()
    # options.execution.shots = 20000
    # session= Session(service=service, backend=backend)
    # estimator = Estimator(session=session, options=options)


    ##### CAFQA PEEPS VQE ######

    # vqe_energy, vqe_params = run_vqe(
    #     n_qubits=n_qubits,
    #     coeffs=coeffs,
    #     paulis=paulis,
    #     param_guess=[],
    #     budget=500,
    #     shots=10**4,
    #     mode="no_noisy_sim",
    #     backend=backend,
    #     save_dir=save_dir,
    #     loss_file="vqe_loss.txt",
    #     params_file="vqe_params.txt",
    #     vqe_kwargs=vqe_kwargs
    # )

    # print("----------------------------------------------")
    # print("VQE energy: "+str(vqe_energy))
    # print("----------------------------------------------")

    # cafqa_vqe_energy, cafqa_vqe_params = run_vqe(
    #     n_qubits=n_qubits,
    #     coeffs=coeffs,
    #     paulis=paulis,
    #     param_guess=np.array(cafqa_parameters),
    #     budget=500,
    #     shots=10**4,
    #     mode="no_noisy_sim",
    #     backend=backend,
    #     save_dir=save_dir,
    #     loss_file="cafqa_vqe_loss.txt",
    #     params_file="cafqa_vqe_params.txt",
    #     vqe_kwargs=vqe_kwargs
    # )

    # print("----------------------------------------------")
    # print("CAFQA + VQE energy: "+str(cafqa_vqe_energy))
    # print("----------------------------------------------")
    # print("Actual energy:"+str(get_ref_energy(coeffs, paulis)))
    # print("----------------------------------------------")

    # def get_vals(loss_filename):
        # temps = []
        # f = open(loss_filename)
        # for line in f.readlines():
        #     temps.append(float(line))
        # f.close()
        # return temps

    # vqe_conv = get_vals("vqe_loss.txt")
    # vqe_cafqa_conv = get_vals("cafqa_vqe_loss.txt")

    # plt.plot(vqe_conv, label="VQE")
    # plt.plot(vqe_cafqa_conv, label="CAFQA + VQE")


    ######## QISKIT VQE #########

    ######## Define optimizer ########
    # opt = ADAM(maxiter=15000,lr=0.01) # >>> 1000 circuits
    opt = COBYLA() # Fast, good convergence
    # opt = SciPyOptimizer(method='BFGS') # bad convergence
    # opt = SPSA(maxiter=1000) # > 1000 circuits
    # opt = SLSQP(maxiter=1000) # ~ 1000 circuits BAAAD
    # opt = SciPyOptimizer(method='CG') # bad convergence
    # opt = AQGD(maxiter=1000,eta=0.2) # >>> 1000 circuits
    # opt = GradientDescent(maxiter=1000) # >>> 1000 circuits
    # opt = NFT(maxiter=1000) #wild convergnece.. slow

    ########### Set up VQE and run on backend ###########
    vqe_only = VQE(estimator=estimator,ansatz=qc, optimizer=opt, callback=log.update)
    result = vqe_only.compute_minimum_eigenvalue(operator=Ham)
    vqe_cafqa = VQE(estimator=estimator,ansatz=qc, optimizer=opt, initial_point=cafqa_parameters, callback=log_cafqa.update)
    result_w_cafqa = vqe_cafqa.compute_minimum_eigenvalue(operator=Ham)

    # print('Initial CAFQA guess on VQE: '+str(log_cafqa.values[0]))

    plt.plot(log.values, label="VQE")
    plt.plot(log_cafqa.values, label="CAFQA + VQE")

    plt.plot([get_ref_energy(coeffs, paulis) for i in range(len(log.values))],"-k",label="Actual")
    plt.xlabel("Iter")
    plt.ylabel('Energy')
    plt.legend()
    plt.title("4-site XXZ Model: XX + YY + Hz*ZZ (Hz = "+str(i)+")")
    plt.show()
    # plt.clf()

    ############ print the results ############
    print("VQE energy: "+str(result.eigenvalue.real))
    e_vqe.append(result.eigenvalue.real)
    print("CAFQA + VQE energy: "+str(result_w_cafqa.eigenvalue.real))
    e_cafqa_vqe.append(result_w_cafqa.eigenvalue.real)

    print("Actual energy:"+str(get_ref_energy(coeffs, paulis)))
    e_act.append(get_ref_energy(coeffs, paulis))

    vqes.append(log.values)
    cafqa_vqes.append(log_cafqa.values)

all_vqe = np.zeros((30,60))
all_cafqa_vqe = np.zeros_like(all_vqe)
for i in range(30):
    for j in range(60):
        all_vqe[i,j] = vqes[i][j]
        all_cafqa_vqe[i,j] = cafqa_vqes[i][j]

# plt.plot(h,e_act,"--k",label="Actual")
# plt.plot(h,e_cafqa,"o",label="CAFQA")
# plt.plot(h,e_vqe,"o",label="VQE")
# plt.plot(h,e_cafqa_vqe,"o",label="CAFQA + VQE")
# plt.xlabel('Hz')
plt.plot([get_ref_energy(coeffs, paulis) for i in range(len(np.average(all_cafqa_vqe, axis=0)))],"-k",label="Actual") 
plt.plot(np.average(all_vqe, axis=0),label="VQE (average)")
plt.plot(np.average(all_vqe, axis=0)[-1],'or',label="VQE converged iteration")
plt.plot(np.average(all_cafqa_vqe, axis=0),label="CAFQA + VQE (average)")
plt.plot(np.average(all_cafqa_vqe, axis=0)[-1],'ob',label="CAFQA + VQE converged iteration")
plt.ylabel('Energy')
plt.legend()
# plt.title("4-site XXZ Model: XX + YY + Hz*ZZ (Noisy Sims)")
plt.title("4-site XXZ Model: XX + YY + Hz*ZZ")
plt.show()
plt.savefig("mygraph.png")

# %%
for i in range(len(log.parameters[0])):
    plt.plot(np.array(log.parameters)[:,i],label="x_"+str(i))
plt.legend(loc="upper right")
plt.xlabel("Iter")
plt.ylabel("Parameter Value")
plt.title("Convergence of parameters for VQE only (Hz = "+str(h[0])+")")
#%%
for i in range(len(log_cafqa.parameters[0])):
    plt.plot(np.array(log_cafqa.parameters)[:,i],label="x_"+str(i))
plt.legend(loc="upper right")
plt.xlabel("Iter")
plt.ylabel("Parameter Value")
plt.title("Convergence of parameters for VQE only (Hz = "+str(h[0])+")")
# %%
