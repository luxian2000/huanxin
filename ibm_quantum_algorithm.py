"""
Quantum Algorithm for IBM Real Quantum Backend

This script demonstrates how to run a quantum algorithm on IBM's real quantum computers.
It implements the Deutsch-Jozsa algorithm and includes support for both simulation and real hardware.

Requirements:
    pip install qiskit qiskit-ibm-runtime qiskit-aer qiskit-machine-learning
"""

import os
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime import QiskitRuntimeService, Session, SamplerV2
import warnings

warnings.filterwarnings('ignore')


def setup_ibm_backend():
    """
    Setup IBM Quantum backend connection.
    
    You need to:
    1. Create an IBM Quantum account at https://quantum.ibm.com
    2. Get your API token from the account settings
    3. Save credentials: QiskitRuntimeService.save_account(channel="ibm_quantum", token="YOUR_TOKEN")
    """
    try:
        service = QiskitRuntimeService(channel="ibm_quantum")
        print("✓ IBM Quantum service initialized successfully")
        return service
    except Exception as e:
        print(f"⚠ Could not connect to IBM Quantum: {e}")
        print("  Running in simulation mode instead...")
        return None


def deutsch_jozsa_circuit(oracle_type='balanced'):
    """
    Create a Deutsch-Jozsa algorithm circuit.
    
    The Deutsch-Jozsa algorithm determines if a function is balanced or constant.
    
    Args:
        oracle_type: 'constant' or 'balanced'
        
    Returns:
        QuantumCircuit: The complete circuit
    """
    n_qubits = 3
    qr = QuantumRegister(n_qubits, 'q')
    cr = ClassicalRegister(n_qubits - 1, 'c')
    circuit = QuantumCircuit(qr, cr, name='Deutsch-Jozsa')
    
    # Initialize the output qubit
    circuit.x(qr[n_qubits - 1])
    
    # Apply Hadamard gates to all qubits
    for i in range(n_qubits):
        circuit.h(qr[i])
    
    # Oracle: constant (identity) or balanced (controlled-NOT)
    if oracle_type == 'constant':
        # Constant function: apply identity (do nothing) or global phase
        circuit.id(qr[0])
    elif oracle_type == 'balanced':
        # Balanced function: apply controlled gates
        for i in range(n_qubits - 1):
            circuit.cx(qr[i], qr[n_qubits - 1])
    
    # Apply Hadamard gates to input qubits
    for i in range(n_qubits - 1):
        circuit.h(qr[i])
    
    # Measure input qubits
    for i in range(n_qubits - 1):
        circuit.measure(qr[i], cr[i])
    
    return circuit


def bell_state_circuit():
    """
    Create a Bell state (entanglement) circuit - a simpler example.
    
    Returns:
        QuantumCircuit: Bell state circuit
    """
    circuit = QuantumCircuit(2, 2, name='Bell-State')
    
    # Create entanglement
    circuit.h(0)
    circuit.cx(0, 1)
    
    # Measure
    circuit.measure([0, 1], [0, 1])
    
    return circuit


def variational_quantum_circuit(params):
    """
    Create a simple variational quantum circuit with parameterized gates.
    
    Args:
        params: List of rotation angles
        
    Returns:
        QuantumCircuit: Parameterized circuit
    """
    circuit = QuantumCircuit(2, 2, name='VQC')
    
    # Parameterized rotation gates
    circuit.ry(params[0], 0)
    circuit.ry(params[1], 1)
    
    # Entanglement
    circuit.cx(0, 1)
    
    # More rotations
    circuit.ry(params[2], 0)
    circuit.ry(params[3], 1)
    
    # Measure
    circuit.measure([0, 1], [0, 1])
    
    return circuit


def run_on_simulator(circuit, shots=1024):
    """
    Run circuit on local simulator (AerSimulator).
    
    Args:
        circuit: QuantumCircuit to run
        shots: Number of measurement shots
        
    Returns:
        dict: Results with counts
    """
    print(f"\n📊 Running on Local Simulator ({shots} shots)...")
    print(f"Circuit: {circuit.name}")
    print(f"Qubits: {circuit.num_qubits}, Clbits: {circuit.num_clbits}")
    
    simulator = AerSimulator()
    job = simulator.run(circuit, shots=shots)
    result = job.result()
    counts = result.get_counts(circuit)
    
    print(f"✓ Results: {counts}")
    return counts


def run_on_ibm_backend(circuit, shots=1024, service=None, backend_name='ibm_brisbane'):
    """
    Run circuit on real IBM quantum backend.
    
    Args:
        circuit: QuantumCircuit to run
        shots: Number of measurement shots
        service: QiskitRuntimeService instance
        backend_name: Name of IBM backend (e.g., 'ibm_brisbane', 'ibm_kyoto')
        
    Returns:
        dict: Results with counts
    """
    if service is None:
        print("⚠ IBM Quantum service not available. Running on simulator instead.")
        return run_on_simulator(circuit, shots)
    
    try:
        print(f"\n🚀 Running on IBM Backend: {backend_name} ({shots} shots)...")
        
        # Get available backends
        backends = service.backends()
        print(f"   Available backends: {[b.name for b in backends[:3]]}...")
        
        # Create a session for multiple jobs
        with Session(service=service, backend=backend_name) as session:
            sampler = SamplerV2(session=session)
            
            # Run the circuit
            job = sampler.run([circuit], shots=shots)
            result = job.result()
            
            # Extract results
            if hasattr(result, '__getitem__'):
                counts = result[0].data.meas.get_counts()
            else:
                counts = result.get_counts(0)
            
            print(f"✓ Job ID: {job.job_id()}")
            print(f"✓ Results: {counts}")
            return counts
            
    except Exception as e:
        print(f"⚠ Error running on IBM backend: {e}")
        print("  Falling back to simulator...")
        return run_on_simulator(circuit, shots)


def main():
    """Main execution function."""
    print("=" * 60)
    print("IBM Quantum Algorithm Runner")
    print("=" * 60)
    
    # Setup
    service = setup_ibm_backend()
    
    # Example 1: Simple Bell State
    print("\n" + "=" * 60)
    print("Example 1: Bell State (Entanglement)")
    print("=" * 60)
    bell_circuit = bell_state_circuit()
    print(bell_circuit)
    run_on_simulator(bell_circuit)
    # To run on real hardware (requires IBM account):
    # run_on_ibm_backend(bell_circuit, service=service, backend_name='ibm_brisbane')
    
    # Example 2: Deutsch-Jozsa Algorithm
    print("\n" + "=" * 60)
    print("Example 2: Deutsch-Jozsa Algorithm (Constant Function)")
    print("=" * 60)
    dj_constant = deutsch_jozsa_circuit('constant')
    print(dj_constant)
    run_on_simulator(dj_constant)
    
    print("\n" + "=" * 60)
    print("Example 3: Deutsch-Jozsa Algorithm (Balanced Function)")
    print("=" * 60)
    dj_balanced = deutsch_jozsa_circuit('balanced')
    print(dj_balanced)
    run_on_simulator(dj_balanced)
    
    # Example 3: Variational Quantum Circuit
    print("\n" + "=" * 60)
    print("Example 4: Variational Quantum Circuit")
    print("=" * 60)
    params = [0.5, 1.2, 0.3, 2.1]
    vqc = variational_quantum_circuit(params)
    print(vqc)
    run_on_simulator(vqc)
    
    print("\n" + "=" * 60)
    print("✓ All examples completed!")
    print("=" * 60)
    print("\nTo run on real IBM quantum hardware:")
    print("1. Create account at https://quantum.ibm.com")
    print("2. Save your API token: QiskitRuntimeService.save_account(...)")
    print("3. Uncomment the run_on_ibm_backend() calls in the code")


if __name__ == "__main__":
    main()
