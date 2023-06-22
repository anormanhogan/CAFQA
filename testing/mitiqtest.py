#%%
from mitiq.benchmarks import generate_rb_circuits
from mitiq.zne import execute_with_zne
from mitiq.zne.scaling import (
    fold_gates_at_random,
    fold_global,
    fold_all
)
from mitiq.zne.inference import LinearFactory, RichardsonFactory
from mitiq import (
    Calibrator,
    Settings,
    execute_with_mitigation,
    MeasurementResult,
)

from test import *
from qiskit.providers.fake_provider import FakeJakarta  # Fake (simulated) QPU

### Noise Model ###
import qiskit_aer.noise as noise
from qiskit.providers.aer import AerSimulator
from qiskit_ibm_runtime import QiskitRuntimeService, Estimator, Session
from qiskit_ibm_provider import IBMProvider
from qiskit.quantum_info import Statevector

# Save account credentials.
IBMProvider.save_account(instance="ibm-q/open/main",token='21fc925c651c9dfa502b200726935702e51579ecd19c0644e284df58209412fb3e994ee059d1a93d9ac8ad79ce4fe753d2522f12a0a2a8265ac66a501609c757', overwrite=True)

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
noisy_sim = AerSimulator(method='statevector',noise_model=noise_model)
backend = 'ibmq_qasm_simulator'

coeffs, paulis = XYmodel(0.1)

# Define hamiltonian

Ham = SparsePauliOp(paulis,coeffs = coeffs).to_matrix()
# print(Ham)
print("GS of H: "+str(np.linalg.eigh(Ham)[0][0]))

# n_qubits = 2
# depth_circuit = 100
# shots = 10 ** 4
# circuit = generate_rb_circuits(n_qubits, depth_circuit,return_type="qiskit")[0]
# circuit.measure_all()
# print(len(circuit))

param = [0,1,3,2,0,0,1,2]
circuit = QuantumCircuit(4,4)
add_ansatz(circuit,ansatz,param,2)
depth_circuit = len(circuit)
circuit.measure_all()
shots = 10**4

def execute_circuit(circuit):
    """Execute the input circuit and return the expectation value of |00..0><00..0|"""
    noisy_backend = noisy_sim
    noisy_result = noisy_backend.run(circuit, shots=shots).result()
    noisy_counts = noisy_result.get_counts(circuit)
    noisy_counts = { k.replace(" 0000",""):v for k, v in noisy_counts.items()}
    # print(noisy_counts)
    keys = []
    for i in range(2**4):
        b = bin(i)[2:]
        l = len(b)
        keys.append(str(0) * (4 - l) + b)
    # print(keys)
    noisy_counts_sorted = []
    for i in keys:
        try:
            noisy_counts_sorted.append(noisy_counts[i]/shots)
        except:
            noisy_counts_sorted.append(0)
    # print(noisy_counts_sorted)
    state = Statevector(noisy_counts_sorted)
    # print(state)
    noisy_expectation_value = state.conjugate().inner(Ham.dot(state))
    print(noisy_expectation_value)
    return noisy_expectation_value

mitigated = execute_with_zne(circuit, execute_circuit, factory=LinearFactory([1, 3, 5]))
unmitigated = execute_circuit(circuit)
ideal = get_ref_energy(coeffs, paulis) 

print("ideal = \t \t",ideal)
print("unmitigated = \t \t", "{:.5f}".format(unmitigated))
print("mitigated = \t \t", "{:.5f}".format(mitigated))

def execute_calibration(qiskit_circuit):
    """Execute the input circuits and return the measurement results."""
    noisy_backend = FakeJakarta()
    noisy_result = noisy_backend.run(qiskit_circuit, shots=shots).result()
    noisy_counts = noisy_result.get_counts(qiskit_circuit)
    noisy_counts = { k.replace(" ",""):v for k, v in noisy_counts.items()}
    measurements = MeasurementResult.from_counts(noisy_counts)
    return measurements

RBSettings = Settings(
    benchmarks=[
        {
            "circuit_type": "rb",
            "num_qubits": 2,
            "circuit_depth": int(depth_circuit / 2),
        },
    ],
    strategies=[
        {
            "technique": "zne",
            "scale_noise": fold_global,
            "factory": RichardsonFactory([1.0, 2.0, 3.0]),
        },
        {
            "technique": "zne",
            "scale_noise": fold_global,
            "factory": RichardsonFactory([1.0, 3.0, 5.0]),
        },
        {
            "technique": "zne",
            "scale_noise": fold_global,
            "factory": LinearFactory([1.0, 2.0, 3.0]),
        },
        {
            "technique": "zne",
            "scale_noise": fold_global,
            "factory": LinearFactory([1.0, 3.0, 5.0]),
        },

        {
            "technique": "zne",
            "scale_noise": fold_gates_at_random,
            "factory": RichardsonFactory([1.0, 2.0, 3.0]),
        },
        {
            "technique": "zne",
            "scale_noise": fold_gates_at_random,
            "factory": RichardsonFactory([1.0, 3.0, 5.0]),
        },
        {
            "technique": "zne",
            "scale_noise": fold_gates_at_random,
            "factory": LinearFactory([1.0, 2.0, 3.0]),
        },
        {
            "technique": "zne",
            "scale_noise": fold_gates_at_random,
            "factory": LinearFactory([1.0, 3.0, 5.0]),
        },

        {
            "technique": "zne",
            "scale_noise": fold_all,
            "factory": RichardsonFactory([1.0, 2.0, 3.0]),
        },
        {
            "technique": "zne",
            "scale_noise": fold_all,
            "factory": RichardsonFactory([1.0, 3.0, 5.0]),
        },
        {
            "technique": "zne",
            "scale_noise": fold_all,
            "factory": LinearFactory([1.0, 2.0, 3.0]),
        },
        {
            "technique": "zne",
            "scale_noise": fold_all,
            "factory": LinearFactory([1.0, 3.0, 5.0]),
        },
        
    ],
)

cal = Calibrator(execute_calibration, frontend="qiskit", settings=RBSettings)
cal.run(log=True)

calibrated_mitigated=execute_with_mitigation(circuit, execute_circuit, calibrator=cal)
mitigated=execute_with_zne(circuit, execute_circuit, factory=LinearFactory([1, 3, 5]))
unmitigated=execute_circuit(circuit)

print("ideal = \t \t",ideal)
print("unmitigated = \t \t", "{:.5f}".format(unmitigated))
print("mitigated = \t \t", "{:.5f}".format(mitigated))
print("calibrated_mitigated = \t", "{:.5f}".format(calibrated_mitigated))

# %%
