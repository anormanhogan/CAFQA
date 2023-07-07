
#%%
import numpy as np
import scipy as scipy
from itertools import permutations

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

pi = np.pi
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

###########################
sys.path.append("/Users/norman/Documents/GitHub/BayesianOptimization/")

#Bayesian Optimization module
from bayes_opt import BayesianOptimization
from bayes_opt.logger import JSONLogger
from bayes_opt.event import Events

num_qubits = 4
training_points = [0.2,1.2,2.2] #bz
num_tp = len(training_points)
bx = 0
bz = 1 #for final comparisons
j = -1
periodic = False

#Bz tuning 
Bz_spec = np.linspace(0,3,10)

#ansatz repetitions
reps = 2

def remove_duplicates(arr):
    # Create an empty set to keep track of unique elements
    unique_elements = set()
    
    # Create a new list to store the elements without duplicates
    result = []
    
    # Iterate over each element in the array
    for element in arr:
        # Check if the element is not already in the set of unique elements
        if tuple(element) not in unique_elements:
            # Add the element to the set of unique elements
            unique_elements.add(tuple(element))
            
            # Append the element to the result list
            result.append(element)
    
    return np.array(result)

def generate_permutations(n_qubits,pauli,periodic):
    permutations_list = []
    base_string =  pauli + "I" * (n_qubits - 2)
    
    # Generate all permutations of base string
    base_permutations = list(permutations(base_string))
    
    # Append "XX" to each permutation
    for perm in base_permutations:
        perm = ''.join(perm)
        if (pauli in perm) and (perm not in permutations_list):
            permutations_list.append(perm)

    if periodic:
        permutations_list.append(pauli[1]+"I" * (n_qubits - 2) + pauli[0])
    
    return permutations_list

def XYmodel(n_qubits,j,bx,bz,periodic):
    
    xxyy_coeff = np.array([j for i in range(2*(n_qubits-1))])
    bx_coeff = np.array([bx for i in range(n_qubits)])
    bz_coeff = np.array([bz for i in range(n_qubits)])

    xxyy_paulis = generate_permutations(n_qubits,'XX',periodic) + generate_permutations(n_qubits,'YY',periodic)
    xperms = [''.join(perm) for perm in permutations('I'*(n_qubits-1)+'X')]
    zperms = [''.join(perm) for perm in permutations('I'*(n_qubits-1)+'Z')]
    bx_paulis = []
    bz_paulis = []

    for i in range(len(xperms)):
        if xperms[i] not in bx_paulis:
            bx_paulis.append(xperms[i])
        if zperms[i] not in bz_paulis:
            bz_paulis.append(zperms[i])

    coeffs = [*xxyy_coeff,*bx_coeff,*bz_coeff]
    paulis = [*xxyy_paulis,*bx_paulis,*bz_paulis]

    return coeffs, paulis

def make_vectors(cafqa_parameters):
    qc = QuantumCircuit(num_qubits,num_qubits)
    add_ansatz(qc,ansatz,cafqa_parameters,reps)
    vec = Statevector(qc)
    return vec
    
def make_basis(training_points,bx):
    basis=np.zeros([len(training_points),2**num_qubits],dtype=complex)
    cafqa_parameters = []

    for i in range(len(training_points)):

        print('##############################################')
        print('Beginning EC training with CAFQA procedure for Bz = '+str(training_points[i])+' ('+str(i+1)+'/'+str(len(training_points))+') ...')
        print('##############################################')
        
        ####### Generate Hamiltonian coefficients and Paulis
        coeffs, paulis = XYmodel(num_qubits,j,bx,i,periodic)

        ####### Get num of parameters:
        qc, num_param = ansatz(num_qubits,reps)

        ######## Bounded region of parameter space 
        pbounds = {'x_'+str(i): (0,3) for i in range(num_param)}
        
        def black_box_function(**params):

            p_vec = []

            ### Force discrete parameters ###
            for p in params.values():
                p_vec.append(int(round(p,0)))
            
            parameters = [p*(pi/2) for p in p_vec]

            vqe_qc = QuantumCircuit(num_qubits)

            ### Generates the ansatz with parameters filled ###
            add_ansatz(vqe_qc, ansatz, parameters, reps)

            ### Transforms all gates to Clifford only ###
            vqe_qc_trans = transform_to_allowed_gates(vqe_qc)
            stim_qc = qiskit_to_stim(vqe_qc_trans)
            
            ### Efficiently computing expectation value of Clifford circuits
            sim = stim.TableauSimulator()
            sim.do_circuit(stim_qc)
            pauli_expect = [sim.peek_observable_expectation(stim.PauliString(p)) for p in paulis]

            #Compute loss (energy)
            loss = np.dot(coeffs, pauli_expect)

            # BO *maximizes*, so return the negative of the loss (energy)
            return -loss

        if num_param < num_qubits**2:
            add_it = num_qubits**2 - num_param
        else:
            add_it = 0 

        optimizer = BayesianOptimization(
            f = black_box_function,
            pbounds=pbounds,
            verbose=2, # verbose = 1 prints only when a maximum is observed, verbose = 0 is silent
            random_state=0,
        )
        optimizer.maximize(
            init_points=num_param,
            n_iter=add_it, #Additional iterations
        )
        
        ####### Print the converged CAFQA energy
        print("CAFQA ground energy: "+str(-optimizer.max['target']))

        #Generates the Hamiltonian as a matrix
        Ham = SparsePauliOp(paulis,coeffs = coeffs).to_matrix()
        print("Actual ground energy: "+str(np.linalg.eigh(Ham)[0][0]))

        ########## Save the optimal CAFQA parameters #########
        cafqa_parameters.append(np.array([int(round(p,0))*(pi/2) for p in optimizer.max['params'].values()]))

        print("Converged CAFQA parameters: ",cafqa_parameters[i])
        basis[i] = make_vectors(cafqa_parameters[i])

    dupes = len(cafqa_parameters) - len(set([tuple(param) for param in cafqa_parameters]))

    if dupes != 0:
        print("**********************************************************************************************")
        print("**********************************************************************************************")
        print("WARNING: There are "+str(dupes)+" duplicate CAFQA basis vectors. Duplicates will be discarded.")
        print("**********************************************************************************************")
        print("**********************************************************************************************")
        basis = remove_duplicates(basis)

    return basis, cafqa_parameters

def cafqa_energy_spec(Bz_spec,bx):
    cafqa_energy = []
    cafqa_GS = np.zeros((len(Bz_spec),2**num_qubits),dtype=complex)
    cafqa_parameters = []
    for i,Bz in enumerate(Bz_spec):

        print('##############################################')
        print('Beginning CAFQA procedure for Bz = '+str(Bz_spec[i])+' ('+str(i+1)+'/'+str(len(Bz_spec))+') ...')
        print('##############################################')
        
        ####### Generate Hamiltonian coefficients and Paulis
        coeffs, paulis = XYmodel(num_qubits,j,bx,Bz,periodic)

        ####### Get num of parameters:
        qc, num_param = ansatz(num_qubits,reps)

        ######## Bounded region of parameter space 
        pbounds = {'x_'+str(i): (0,3) for i in range(num_param)}
        
        def black_box_function(**params):

            p_vec = []

            ### Force discrete parameters ###
            for p in params.values():
                p_vec.append(int(round(p,0)))
            
            parameters = [p*(pi/2) for p in p_vec]

            vqe_qc = QuantumCircuit(num_qubits)

            ### Generates the ansatz with parameters filled ###
            add_ansatz(vqe_qc, ansatz, parameters, reps)

            ### Transforms all gates to Clifford only ###
            vqe_qc_trans = transform_to_allowed_gates(vqe_qc)
            stim_qc = qiskit_to_stim(vqe_qc_trans)
            
            ### Efficiently computing expectation value of Clifford circuits
            sim = stim.TableauSimulator()
            sim.do_circuit(stim_qc)
            pauli_expect = [sim.peek_observable_expectation(stim.PauliString(p)) for p in paulis]

            #Compute loss (energy)
            loss = np.dot(coeffs, pauli_expect)

            # BO *maximizes*, so return the negative of the loss (energy)
            return -loss

        if num_param < num_qubits**2:
            add_it = num_qubits**2 - num_param
        else:
            add_it = 0 

        optimizer = BayesianOptimization(
            f = black_box_function,
            pbounds=pbounds,
            verbose=2, # verbose = 1 prints only when a maximum is observed, verbose = 0 is silent
            random_state=0,
        )
        optimizer.maximize(
            init_points=num_param,
            n_iter=add_it, #Additional iterations
        )
        
        ####### Print the converged CAFQA energy
        print("CAFQA ground energy: "+str(-optimizer.max['target']))

        #Generates the Hamiltonian as a matrix
        Ham = SparsePauliOp(paulis,coeffs = coeffs).to_matrix()
        print("Actual ground energy: "+str(np.linalg.eigh(Ham)[0][0]))

        ########## Save the CAFQA energy #########
        cafqa_energy.append(-optimizer.max['target'])
        cafqa_parameters.append(np.array([int(round(p,0))*(pi/2) for p in optimizer.max['params'].values()]))

        qc = QuantumCircuit(num_qubits,num_qubits)
        add_ansatz(qc,ansatz,cafqa_parameters[i],reps)
        vec = Statevector(qc)
        cafqa_GS[i]=vec

    return cafqa_energy, cafqa_GS

def do_EC(Ham, basis):

    nvecs = len(basis)
    small_Ham = np.zeros([nvecs,nvecs],dtype='complex')
    overlap = np.zeros_like(small_Ham)

    for i in range(nvecs):

        ui = basis[i,:]

        for j in range(i,nvecs):
            uj = basis[j,:]
            
            small_Ham[i,j] = np.conjugate(np.transpose(ui)) @ Ham @ uj

            if not i == j:
                small_Ham[j,i] = np.conjugate(small_Ham[i,j])

            overlap[i,j] = np.dot(np.conjugate(np.transpose(ui)),uj)

            if not i == j:
                overlap[j,i] = np.conjugate(overlap[i,j])

    return small_Ham, overlap

def get_new_evals(Ham,basis):
   
    small_Ham, overlap = do_EC(Ham,basis)
    evals, evecs = scipy.linalg.eigh(small_Ham,overlap)

    nvecs = len(basis)
    
    gs_vec = np.zeros_like(basis[0])
    new_evals=np.zeros([nvecs])
    for k in range(nvecs):
        fullvec = np.zeros_like(basis[0])
        for l in range(nvecs):
            fullvec += evecs[l,k] * basis[l]
        if k == 0: gs_vec = fullvec   # ground state vector is stored for fidelity calculations
        energy = np.transpose(fullvec) @ Ham @fullvec
        new_evals[k] = -np.abs(energy)

    return new_evals, gs_vec

def plot_energy_spec(Bz_spec,n_qubits,j,bx):
    energies = np.zeros((len(Bz_spec),2**n_qubits))
    for i,Bz in enumerate(Bz_spec):
        coeffs,paulis = XYmodel(n_qubits,j,bx,Bz,periodic)
        Ham = SparsePauliOp(paulis,coeffs = coeffs).to_matrix()

        energies[i],evecs = scipy.linalg.eigh(Ham)
        energies[i] = energies[i][::-1]

    increment = 1/(2**n_qubits)
    alph = increment
    for i in range(2**n_qubits):
        fig = plt
        if i == 2**n_qubits-1:
            fig.plot(Bz_spec,energies[:,i],'b',alpha=alph, label="Ground state")
            flag = 0
            for point in training_points:
                if flag == 0:
                    fig.plot([point,point],[max(energies.flatten()),min(energies.flatten())],'-r',label="Training points")
                    flag = 1
                else:
                    fig.plot([point,point],[max(energies.flatten()),min(energies.flatten())],'-r',label='_nolegend_')
        else:
            fig.plot(Bz_spec,energies[:,i],'k',alpha=alph,label='_nolegend_')

        alph+=increment
        fig.xlabel("Bz")
        fig.ylabel("Energy")
        if periodic:
            fig.title(str(n_qubits)+" site periodic XY model: Bx="+str(bx))
        else:
            fig.title(str(n_qubits)+" site XY model: Bx="+str(bx))
        fig.legend()
    return fig

def GS_comparisons(Bz_spec,basis,bx,return_GS):
    
    cafqa_energies, cafqa_GS = cafqa_energy_spec(Bz_spec,bx)

    EC_energies = np.zeros(len(Bz_spec))
    EC_GS = np.zeros((len(Bz_spec),2**num_qubits),dtype=complex)
    exact_energies = np.zeros(len(Bz_spec))
    exact_GS = np.zeros((len(Bz_spec),2**num_qubits),dtype=complex)

    for i,Bz in enumerate(Bz_spec):

        print('##############################################')
        print('Beginning EC procedure for Bz = '+str(Bz_spec[i])+' ('+str(i+1)+'/'+str(len(Bz_spec))+') ...')
        print('##############################################')

        coeffs,paulis = XYmodel(num_qubits,j,bx,Bz,periodic)
        Ham = SparsePauliOp(paulis,coeffs = coeffs).to_matrix()

        evals,EC_GS[i] = get_new_evals(Ham,basis)
        EC_energies[i] = evals[0]

        print("EC ground energy: "+str(EC_energies[0]))

        evals, evecs = scipy.linalg.eigh(Ham)
        exact_energies[i] = evals[0]
        exact_GS[i] = evecs[0]


    cafqa_fid = [np.absolute(np.dot(cafqa_GS[i],exact_GS[i])**2) for i in range(len(Bz_spec))]
    EC_fid = [np.absolute(np.dot(EC_GS[i],exact_GS[i])**2) for i in range(len(Bz_spec))]

    if return_GS:
        data = {"CAFQA energy":cafqa_energies,"CAFQA gs":cafqa_GS,"CAFQA fidelity":cafqa_fid,"EC energy":EC_energies,"EC gs":EC_GS,"EC fidelity":EC_fid,"exact energy":exact_energies,"exact gs":exact_GS}
    else:
        data = {"CAFQA energy":cafqa_energies,"CAFQA fidelity":cafqa_fid,"EC energy":EC_energies,"EC fidelity":EC_fid,"exact energy":exact_energies}
    return data

def plot_comparisons(Bz_spec,basis,bx):

    data = GS_comparisons(Bz_spec,basis,bx,False)

    plt.plot(Bz_spec,data["exact energy"],'-k',label="Exact")
    plt.plot(Bz_spec,data["CAFQA energy"],'--o',label="CAFQA")
    plt.plot(Bz_spec,data["EC energy"],'--o',label="EC with "+str(len(basis))+" unique training pts")
    plt.xlabel("Bz")
    plt.ylabel("Energy")
    if periodic:
        plt.title(str(num_qubits)+" site periodic XY model: Bx="+str(bx))
    else:
        plt.title(str(num_qubits)+" site XY model: Bx="+str(bx))
    plt.legend()

    plt.savefig("XY_energy_comparisons.png")
    plt.show()
    plt.clf()

    plt.plot(Bz_spec,data["CAFQA fidelity"],'--o',label="CAFQA")
    plt.plot(Bz_spec,data["EC fidelity"],'--o',label="EC with "+str(len(basis))+" unique training pts")
    plt.xlabel("Bz")
    plt.ylabel("Fidelity")
    if periodic:
        plt.title("GS fidelity for "+str(num_qubits)+" site periodic XY model: Bx="+str(bx))
    else:
        plt.title("GS fidelity for "+str(num_qubits)+" site XY model: Bx="+str(bx))
    plt.legend()

    plt.savefig("XY_fidelity_comparisons.png")
    plt.show()

    return plt, data

def pretty_print(basis):

    for vec in basis:
        nbits = int(np.log2(len(vec)))
        
        formatstr = "{0:>0" + str(nbits) + "b}"
        
        ix=-1
        for x in vec:
            ix += 1
            if abs(x) < 1e-8:
                continue
            
            print(formatstr.format(ix),": ",x)
        print('----------------------------')
    


# qc = QuantumCircuit(num_qubits,num_qubits)
# qc,n_p = ansatz(num_qubits,reps)
# params = np.random.randint(4, size=n_p)
# params = [pi*p/2 for p in params]
# add_ansatz(qc,ansatz,params,reps)
# qc.draw()
# dag = circuit_to_dag(qc)
# dag.draw()
# for node in dag.op_nodes():
#     print(node.name)
# trans_qc = transform_to_allowed_gates(qc)
# trans_qc.draw()

print("*********\nPlotting Energies from Exact Diagonization\n*********\n\n")
fig = plot_energy_spec(Bz_spec,num_qubits,j,bx)
fig.show()

print("*********\nCreating Basis for EC\n*********\n\n")
basis, cafqa_params = make_basis(training_points,bx)

print("*********\nGenerating Comparison Plots\n*********\n\n")
plot, data = plot_comparisons(Bz_spec,basis,bx)

#%%

coeffs,paulis = XYmodel(num_qubits,j,bx,bz)
Ham = SparsePauliOp(paulis,coeffs = coeffs).to_matrix()
new_evals, gs_vec = get_new_evals(Ham,basis)

print("EC GS Energy: ", new_evals[0])
print("Actual GS Energy: ",scipy.linalg.eigh(Ham)[0][0])


# %%
