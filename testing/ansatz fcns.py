def ansatz(n_qubits, repetitions):
    """
    XY ansatz with full entanglement.
    n_qubits (Int): Number of qubits in circuit.
    repetitions (Int): # ansatz repetitions.

    Returns: 
    (QuantumCircuit, Int) (ansatz, #parameters).
    """
    ansatz = QuantumCircuit(n_qubits,n_qubits)
    params_per_rep = 3*(n_qubits-1)+1

    for i in range (repetitions):
        if i == 0:
            paramvec = np.array([Parameter("x_"+str(p)) for p in range(params_per_rep)])
        else:
            newparams = np.array([Parameter("x_"+str(p + i*params_per_rep)) for p in range(params_per_rep)])
            paramvec = np.append(paramvec,newparams)

        count = 0

        for qubit in range(n_qubits-1):
            ansatz.rxx(paramvec[i*params_per_rep + count],qubit,qubit+1)
            count+=1

        for qubit in range(n_qubits-1):
            ansatz.ryy(paramvec[i*params_per_rep + count],qubit,qubit+1)
            count+=1
        
        for qubit in range(n_qubits-1):
            if qubit == 0:
                ansatz.h(qubit)
                ansatz.rz(paramvec[i*params_per_rep + count],qubit)
                ansatz.h(qubit)
                count+=1
                ansatz.h(qubit+1)
                ansatz.rz(paramvec[i*params_per_rep + count],qubit+1)
                ansatz.h(qubit+1)
                count+=1
            else:
                ansatz.h(qubit+1)
                ansatz.rz(paramvec[i*params_per_rep + count],qubit+1)
                ansatz.h(qubit+1)
                count+=1

    num_params_ansatz = len(ansatz.parameters)
    ansatz = ansatz.decompose(gates_to_decompose=['rxx','ryy'])
    return ansatz, num_params_ansatz