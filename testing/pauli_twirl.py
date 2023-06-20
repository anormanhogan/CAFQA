import numpy as np
from itertools import combinations

from qiskit import *
# Need gate classes for generating the Pauli twirling sets
from qiskit.circuit.library import (IGate, XGate, YGate, ZGate,
                                    CXGate, CZGate, ECRGate, iSwapGate)

# Operator class
from qiskit.quantum_info import Operator

# Classes for building up a directed-acyclic graph (DAG) structure
from qiskit.circuit import QuantumRegister
from qiskit.dagcircuit import DAGCircuit
# Transpiler stuff neded to make a pass and passmanager
from qiskit.transpiler import PassManager
from qiskit.transpiler.basepasses import TransformationPass
from qiskit.transpiler.passes import Optimize1qGatesDecomposition
from qiskit.providers.fake_provider import FakeVigo
from qiskit.opflow import Z, X, I, Y, CX


def generate_pauli_twirling_sets(two_qubit_gate):
    """Generate the Pauli twirling sets for a given 2Q gate
    
    Sets are ordered such that gate[0] and gate[1] are pre-roations
    applied to control and target, respectively.  gate[2] and gate[3]
    are post-rotations for control and target, respectively.
    
    Parameters:
        two_qubit_gate (Gate): Input two-qubit gate
        
    Returns:
        list: List of all twirling gate sets
    """
    # Generate 16 element list of Pauli gates, each repeated 4 times
    operator_list = [I, Z, X, Y]*4
    # This is the target unitary to which our twirled circuit should match
    target_unitary = Operator(two_qubit_gate.to_matrix())
    twirling_sets = []
    
    # For every combination in 16 choose 4 make a circuit and look for equivilence
    for gates in combinations(operator_list, 4):
        # Build a circuit for our twirled 2Q gate
        qc = QuantumCircuit(2)
        qc.append(gates[0], [0])
        qc.append(gates[1], [1])
        qc.append(two_qubit_gate, [0, 1])
        qc.append(gates[2], [0])
        qc.append(gates[3], [1])
        
        norm = np.linalg.norm(Operator.from_circuit(qc)-target_unitary)
        
        phase = None
        # If unitaries match we have a phase of zero
        if abs(norm) < 1e-15:
            phase = 0
        # If unitaries differ by a phase of pi, shift by pi
        elif abs(norm-4) < 1e-15:
            phase = np.pi

        if phase is not None:
            qc.global_phase += phase
            # Verify that our twirled circuit is a valid replacement
            assert Operator.from_circuit(qc) == target_unitary
            twirl_set = (gates, phase)
            # Check that set does not already exist
            if twirl_set not in twirling_sets:
                twirling_sets.append(twirl_set)
            
    return twirling_sets

twirling_groups = {} 

for gate in [CX]:
    twirl_set = generate_pauli_twirling_sets(gate)
    twirling_groups['cx'] = twirl_set
    
class PauliTwirling(TransformationPass):
    """Pauli twirl an input circuit.
    """
    def __init__(self, twirling_gate, seed=None):
        """
        Parameters:
            twirling_gate (str): Which gate to twirl
            seed (int): Seed for RNG, should be < 2e32
        """
        super().__init__()
        # This is the target gate to twirl
        self.twirling_gate = twirling_gate
        # Get the twirling set from the dict we generated above
        # This should be repalced by a cached version in practice
        self.twirling_set = twirling_groups[twirling_gate]
        # Length of the twirling set to bound RNG generation
        self.twirling_len = len(self.twirling_set)
        # Seed the NumPy RNG
        self.rng = np.random.default_rng(seed)

    def run(self, dag):
        """Insert Pauli twirls into input DAG
        
        Parameters:
            dag (DAGCircuit): Input DAG
        
        Returns:
            dag: DAG with twirls added in-place
        """
        for run in dag.collect_runs([self.twirling_gate]):
            for node in run:
                # Generate a random int to specify the twirling gates
                twirl_idx = self.rng.integers(0, self.twirling_len)
                # Get the randomly selected twirling set
                twirl_gates = self.twirling_set[twirl_idx][0]
                twirl_phase = self.twirling_set[twirl_idx][1]
                # Make a small DAG for the twirled circuit we are going to insert
                twirl_dag = DAGCircuit()
                # Add a register of qubits (here always 2Q)
                qreg = QuantumRegister(2)
                twirl_dag.add_qreg(qreg)
                # gate[0] pre-applied to control
                twirl_dag.apply_operation_back(twirl_gates[0], [qreg[0]])
                # gate[1] pre-applied to target
                twirl_dag.apply_operation_back(twirl_gates[1], [qreg[1]])
                # Insert original gate
                twirl_dag.apply_operation_back(node.op, [qreg[0], qreg[1]])
                # gate[2] pre-applied to control
                twirl_dag.apply_operation_back(twirl_gates[2], [qreg[0]])
                # gate[3] pre-applied to target
                twirl_dag.apply_operation_back(twirl_gates[3], [qreg[1]])
                # Add a global phase gate to account for possible phase difference
                twirl_dag.global_phase += twirl_phase
                # Replace the target gate with the twirled version
                dag.substitute_node_with_dag(node, twirl_dag)
        return dag