#!/usr/bin/env python3
"""
Spiking Neural Network Neuron Model
====================================
A single Hodgkin-Huxley neuron that:
1. Takes spike trains from rate-encoded MNIST data
2. Computes I_inj as weighted sum of input spikes
3. Simulates HH dynamics with CORDIC-compatible functions
4. Detects output spikes via threshold crossing
"""

import numpy as np
import matplotlib.pyplot as plt
import os

# ==============================================================================
# CONFIGURATION
# ==============================================================================
SPIKE_THRESHOLD = 0.0  # mV - Membrane potential threshold for spike detection
REFRACTORY_PERIOD = 2.0  # ms - Minimum time between spikes

# ==============================================================================
# FIXED-POINT ARITHMETIC (For hardware compatibility)
# ==============================================================================
N_BITS = 22
F_BITS = 12
SCALE = 1 << F_BITS

def to_fixed_point(f):
    """Convert floating point to fixed point representation"""
    return np.int64(f * SCALE)

def to_float(fp):
    """Convert fixed point to floating point"""
    return fp / SCALE

# ==============================================================================
# CORDIC FUNCTIONS (Hardware-friendly exponential)
# ==============================================================================
def cordic_mult(x_fp, y_fp):
    """Fixed-point multiplication"""
    return np.int64((np.int64(x_fp) * np.int64(y_fp)) >> F_BITS)

def cordic_div(x_fp, y_fp):
    """Fixed-point division"""
    if y_fp == 0:
        return np.iinfo(np.int64).max
    return np.int64((np.int64(x_fp) << F_BITS) // np.int64(y_fp))

def cordic_exp(x_fp):
    """CORDIC exponential function"""
    x = to_float(x_fp)
    if x > 10:
        return to_fixed_point(np.exp(10))
    if x < -20:
        return 0
    return to_fixed_point(np.exp(x))

# Saturating arithmetic
def sat_add(a, b):
    """Saturating addition"""
    return np.clip(np.int64(a) + np.int64(b), 
                   np.iinfo(np.int64).min, np.iinfo(np.int64).max)

def sat_sub(a, b):
    """Saturating subtraction"""
    return np.clip(np.int64(a) - np.int64(b),
                   np.iinfo(np.int64).min, np.iinfo(np.int64).max)

# ==============================================================================
# HODGKIN-HUXLEY MODEL PARAMETERS
# ==============================================================================
# All parameters in fixed-point
Cm_fp = to_fixed_point(1.0)        # Membrane capacitance (µF/cm²)
g_Na_fp = to_fixed_point(120.0)    # Sodium conductance (mS/cm²)
g_K_fp = to_fixed_point(36.0)      # Potassium conductance (mS/cm²)
g_L_fp = to_fixed_point(0.3)       # Leak conductance (mS/cm²)
E_Na_fp = to_fixed_point(50.0)     # Sodium reversal potential (mV)
E_K_fp = to_fixed_point(-77.0)     # Potassium reversal potential (mV)
E_L_fp = to_fixed_point(-54.4)     # Leak reversal potential (mV)

# ==============================================================================
# GATING VARIABLE FUNCTIONS
# ==============================================================================
def alpha_n(V_fp):
    """Potassium activation rate"""
    V = to_float(V_fp)
    if abs(V + 55) < 0.1:
        return to_fixed_point(0.1)
    return to_fixed_point(0.01 * (V + 55) / (1 - np.exp(-(V + 55) / 10)))

def beta_n(V_fp):
    """Potassium deactivation rate"""
    V = to_float(V_fp)
    return to_fixed_point(0.125 * np.exp(-(V + 65) / 80))

def alpha_m(V_fp):
    """Sodium activation rate"""
    V = to_float(V_fp)
    if abs(V + 40) < 0.1:
        return to_fixed_point(1.0)
    return to_fixed_point(0.1 * (V + 40) / (1 - np.exp(-(V + 40) / 10)))

def beta_m(V_fp):
    """Sodium deactivation rate"""
    V = to_float(V_fp)
    return to_fixed_point(4.0 * np.exp(-(V + 65) / 18))

def alpha_h(V_fp):
    """Sodium inactivation rate"""
    V = to_float(V_fp)
    return to_fixed_point(0.07 * np.exp(-(V + 65) / 20))

def beta_h(V_fp):
    """Sodium recovery rate"""
    V = to_float(V_fp)
    return to_fixed_point(1.0 / (1 + np.exp(-(V + 35) / 10)))

# ==============================================================================
# IONIC CURRENTS
# ==============================================================================
def I_Na(V_fp, m_fp, h_fp):
    """Sodium current"""
    m3 = cordic_mult(cordic_mult(m_fp, m_fp), m_fp)
    return cordic_mult(cordic_mult(g_Na_fp, m3), 
                      cordic_mult(h_fp, sat_sub(V_fp, E_Na_fp)))

def I_K(V_fp, n_fp):
    """Potassium current"""
    n4 = cordic_mult(cordic_mult(n_fp, n_fp), cordic_mult(n_fp, n_fp))
    return cordic_mult(g_K_fp, cordic_mult(n4, sat_sub(V_fp, E_K_fp)))

def I_L(V_fp):
    """Leak current"""
    return cordic_mult(g_L_fp, sat_sub(V_fp, E_L_fp))

# ==============================================================================
# HODGKIN-HUXLEY ODE SOLVER
# ==============================================================================
def hh_step(V_fp, m_fp, h_fp, n_fp, I_inj_fp, dt):
    """
    Single step of HH model using Euler method.
    
    Args:
        V_fp, m_fp, h_fp, n_fp: Current state (fixed-point)
        I_inj_fp: Injected current (fixed-point)
        dt: Time step in ms
        
    Returns:
        Updated state (V, m, h, n) in fixed-point
    """
    dt_fp = to_fixed_point(dt)
    
    # Calculate ionic currents
    I_na = I_Na(V_fp, m_fp, h_fp)
    I_k = I_K(V_fp, n_fp)
    I_l = I_L(V_fp)
    
    # Membrane potential derivative
    dVdt = cordic_div(sat_sub(I_inj_fp, sat_add(I_na, sat_add(I_k, I_l))), Cm_fp)
    
    # Gating variable derivatives
    an = alpha_n(V_fp)
    bn = beta_n(V_fp)
    dndt = sat_sub(cordic_mult(an, sat_sub(to_fixed_point(1.0), n_fp)), 
                   cordic_mult(bn, n_fp))
    
    am = alpha_m(V_fp)
    bm = beta_m(V_fp)
    dmdt = sat_sub(cordic_mult(am, sat_sub(to_fixed_point(1.0), m_fp)), 
                   cordic_mult(bm, m_fp))
    
    ah = alpha_h(V_fp)
    bh = beta_h(V_fp)
    dhdt = sat_sub(cordic_mult(ah, sat_sub(to_fixed_point(1.0), h_fp)), 
                   cordic_mult(bh, h_fp))
    
    # Euler integration
    V_new = sat_add(V_fp, cordic_mult(dVdt, dt_fp))
    m_new = sat_add(m_fp, cordic_mult(dmdt, dt_fp))
    h_new = sat_add(h_fp, cordic_mult(dhdt, dt_fp))
    n_new = sat_add(n_fp, cordic_mult(dndt, dt_fp))
    
    # Clamp gating variables to [0, 1]
    m_new = np.clip(m_new, to_fixed_point(0.0), to_fixed_point(1.0))
    h_new = np.clip(h_new, to_fixed_point(0.0), to_fixed_point(1.0))
    n_new = np.clip(n_new, to_fixed_point(0.0), to_fixed_point(1.0))
    
    return V_new, m_new, h_new, n_new

# ==============================================================================
# SPIKE DETECTION
# ==============================================================================
def detect_spike(V_current, V_previous, threshold=SPIKE_THRESHOLD):
    """
    Detect if a spike occurred by checking threshold crossing.
    
    Args:
        V_current: Current membrane potential (mV)
        V_previous: Previous membrane potential (mV)
        threshold: Spike threshold (mV)
        
    Returns:
        Boolean indicating if spike occurred
    """
    return (V_previous <= threshold) and (V_current > threshold)

# ==============================================================================
# MAIN NEURON SIMULATION
# ==============================================================================
def simulate_neuron(spike_train, weights=None, dt=0.01, verbose=True):
    """
    Simulate HH neuron with spike train input.
    
    Args:
        spike_train: Binary array of shape (timesteps, n_inputs)
        weights: Synaptic weights array of shape (n_inputs,)
        dt: Time step in ms
        verbose: Print progress
        
    Returns:
        Dictionary with simulation results
    """
    num_timesteps, num_inputs = spike_train.shape
    
    # Initialize weights if not provided (uniform)
    if weights is None:
        # Default: small uniform weights
        weights = np.ones(num_inputs) * 0.02  # 0.02 µA/cm² per synapse
    
    # Convert weights to fixed-point
    weights_fp = np.array([to_fixed_point(w) for w in weights])
    
    # Initialize neuron state
    V_fp = to_fixed_point(-65.0)  # Resting potential
    m_fp = to_fixed_point(0.05)   # Sodium activation
    h_fp = to_fixed_point(0.6)    # Sodium inactivation  
    n_fp = to_fixed_point(0.32)   # Potassium activation
    
    # Storage arrays
    V_trace = np.zeros(num_timesteps)
    m_trace = np.zeros(num_timesteps)
    h_trace = np.zeros(num_timesteps)
    n_trace = np.zeros(num_timesteps)
    I_inj_trace = np.zeros(num_timesteps)
    output_spikes = np.zeros(num_timesteps, dtype=bool)
    
    last_spike_time = -np.inf
    
    if verbose:
        print(f"Starting simulation: {num_timesteps} timesteps, dt={dt}ms")
        print(f"Input dimensions: {num_inputs} synapses")
        print(f"Weight range: [{np.min(weights):.3f}, {np.max(weights):.3f}] µA/cm²")
    
    for t in range(num_timesteps):
        if verbose and t % 1000 == 0:
            print(f"  Progress: {t}/{num_timesteps}")
        
        # Calculate injected current as weighted sum of input spikes
        # I_inj(t) = Σ w_i * spike_i(t)
        I_inj_fp = np.dot(weights_fp, spike_train[t])
        I_inj = to_float(I_inj_fp)
        
        # Store previous voltage for spike detection
        V_prev = to_float(V_fp) if t > 0 else -65.0
        
        # Update neuron state
        V_fp, m_fp, h_fp, n_fp = hh_step(V_fp, m_fp, h_fp, n_fp, I_inj_fp, dt)
        
        # Convert to float for storage
        V = to_float(V_fp)
        m = to_float(m_fp)
        h = to_float(h_fp)
        n = to_float(n_fp)
        
        # Spike detection with refractory period
        if detect_spike(V, V_prev, SPIKE_THRESHOLD):
            if (t * dt - last_spike_time) > REFRACTORY_PERIOD:
                output_spikes[t] = True
                last_spike_time = t * dt
        
        # Store traces
        V_trace[t] = V
        m_trace[t] = m
        h_trace[t] = h
        n_trace[t] = n
        I_inj_trace[t] = I_inj
    
    # Calculate statistics
    spike_times = np.where(output_spikes)[0]
    num_spikes = len(spike_times)
    
    if verbose:
        print(f"\nSimulation complete:")
        print(f"  Output spikes: {num_spikes}")
        if num_spikes > 0:
            firing_rate = num_spikes / (num_timesteps * dt / 1000)
            print(f"  Firing rate: {firing_rate:.2f} Hz")
        print(f"  V range: [{np.min(V_trace):.1f}, {np.max(V_trace):.1f}] mV")
        print(f"  I_inj range: [{np.min(I_inj_trace):.1f}, {np.max(I_inj_trace):.1f}] µA/cm²")
    
    return {
        'V': V_trace,
        'm': m_trace,
        'h': h_trace,
        'n': n_trace,
        'I_inj': I_inj_trace,
        'spikes': output_spikes,
        'spike_times': spike_times,
        'dt': dt,
        'num_timesteps': num_timesteps
    }

# ==============================================================================
# VISUALIZATION
# ==============================================================================
def plot_neuron_response(results, spike_train=None, save_path='neuron_output.png'):
    """
    Plot comprehensive neuron response.
    """
    time_ms = np.arange(results['num_timesteps']) * results['dt']
    
    fig, axes = plt.subplots(5, 1, figsize=(14, 12), sharex=True)
    
    # 1. Membrane potential with spikes
    ax = axes[0]
    ax.plot(time_ms, results['V'], 'b-', linewidth=0.8, label='V(t)')
    ax.axhline(y=SPIKE_THRESHOLD, color='r', linestyle='--', alpha=0.5, 
               label=f'Spike threshold ({SPIKE_THRESHOLD}mV)')
    ax.axhline(y=-65, color='g', linestyle=':', alpha=0.5, label='Rest (-65mV)')
    
    # Mark spikes
    if len(results['spike_times']) > 0:
        spike_times_ms = results['spike_times'] * results['dt']
        spike_voltages = results['V'][results['spike_times']]
        ax.scatter(spike_times_ms, spike_voltages, color='r', s=50, 
                  marker='o', zorder=5, label=f'Spikes (n={len(results["spike_times"])})')
    
    ax.set_ylabel('Membrane\nPotential (mV)')
    ax.set_title('HH Neuron Processing MNIST Spike Trains')
    ax.legend(loc='upper right', ncol=2, fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([-80, 60])
    
    # 2. Injected current (weighted sum of input spikes)
    ax = axes[1]
    ax.plot(time_ms, results['I_inj'], 'g-', linewidth=0.8)
    ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    ax.set_ylabel('I_inj (µA/cm²)\n[Σ w·spike]')
    ax.set_title('Synaptic Current (Weighted Sum of Input Spikes)')
    ax.grid(True, alpha=0.3)
    
    # 3. Gating variables
    ax = axes[2]
    ax.plot(time_ms, results['m'], 'r-', label='m (Na+ activation)', alpha=0.8)
    ax.plot(time_ms, results['h'], 'b-', label='h (Na+ inactivation)', alpha=0.8)
    ax.plot(time_ms, results['n'], 'g-', label='n (K+ activation)', alpha=0.8)
    ax.set_ylabel('Gating\nVariables')
    ax.set_ylim([0, 1])
    ax.legend(loc='upper right', ncol=3, fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # 4. Input spike count
    if spike_train is not None:
        ax = axes[3]
        spike_counts = np.sum(spike_train, axis=1)
        ax.plot(time_ms, spike_counts, 'k-', linewidth=0.5, alpha=0.7)
        ax.fill_between(time_ms, 0, spike_counts, alpha=0.3)
        ax.set_ylabel('Active\nInputs')
        ax.set_title('Number of Active Input Synapses')
        ax.grid(True, alpha=0.3)
    
    # 5. Output spike raster
    ax = axes[4]
    if len(results['spike_times']) > 0:
        spike_times_ms = results['spike_times'] * results['dt']
        ax.vlines(spike_times_ms, 0, 1, colors='r', linewidth=2, 
                 label=f'{len(results["spike_times"])} output spikes')
        ax.set_ylim([-0.1, 1.1])
    else:
        ax.text(0.5, 0.5, 'No output spikes', ha='center', va='center',
               transform=ax.transAxes, fontsize=12, color='gray')
    ax.set_ylabel('Output\nSpikes')
    ax.set_xlabel('Time (ms)')
    ax.set_yticks([])
    if len(results['spike_times']) > 0:
        ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    
    return fig

# ==============================================================================
# MAIN EXECUTION
# ==============================================================================
def main():
    """
    Main function to run the neuron simulation with MNIST spike data.
    """
    print("="*60)
    print("HH NEURON WITH MNIST SPIKE TRAIN INPUT")
    print("="*60)
    
    # Load spike train from rate encoder output
    spike_file = 'spike_train_verilog.txt'
    if not os.path.exists(spike_file):
        print(f"ERROR: {spike_file} not found!")
        print("Please run the rate encoder first:")
        print("  iverilog -g2012 -o rate_encoder_sim rate_encoder.sv rate_encoder_tb.sv")
        print("  vvp rate_encoder_sim")
        return
    
    # Read spike train
    print(f"\nLoading spike train from {spike_file}...")
    with open(spike_file, 'r') as f:
        lines = f.readlines()
    
    spike_train = np.array([list(map(int, line.strip())) for line in lines], dtype=int)
    print(f"Spike train shape: {spike_train.shape}")
    print(f"  {spike_train.shape[0]} timesteps")
    print(f"  {spike_train.shape[1]} input channels (pixels)")
    
    # Calculate input statistics
    avg_active = np.mean(np.sum(spike_train, axis=1))
    print(f"  Average active inputs per timestep: {avg_active:.1f}")
    
    # Set up synaptic weights
    # For 784 inputs with ~140 active on average, we want total current ~10-20 µA/cm²
    # So each synapse should contribute ~0.1 µA/cm² when active
    num_inputs = spike_train.shape[1]
    
    # Option 1: Uniform weights
    # Adjusted for proper spiking: with ~140 active pixels, we need ~15-20 µA/cm² total
    weights = np.ones(num_inputs) * 0.15  # 0.15 µA/cm² per active synapse
    
    # Option 2: Random weights (uncomment to use)
    # weights = np.random.uniform(0.05, 0.15, num_inputs)
    
    # Option 3: Gaussian weights centered on image (uncomment to use)
    # center = num_inputs // 2
    # weights = np.array([0.2 * np.exp(-((i-center)**2)/(2*100**2)) for i in range(num_inputs)])
    
    print(f"\nSynaptic weights:")
    print(f"  Total synapses: {num_inputs}")
    print(f"  Weight per synapse: {np.mean(weights):.3f} µA/cm²")
    print(f"  Expected I_inj range: [0, {avg_active * np.mean(weights):.1f}] µA/cm²")
    
    # Run simulation
    print("\n" + "="*60)
    print("RUNNING SIMULATION...")
    print("="*60)
    
    # Use subset for faster testing (e.g., first 1000 timesteps)
    timesteps_to_simulate = min(1000, spike_train.shape[0])
    
    results = simulate_neuron(
        spike_train[:timesteps_to_simulate],
        weights=weights,
        dt=0.0625,  # 0.01 ms timestep
        verbose=True
    )
    
    # Save output spikes
    output_file = 'output_spikes.txt'
    with open(output_file, 'w') as f:
        f.write(f"# HH Neuron Output Spikes\n")
        f.write(f"# Threshold: {SPIKE_THRESHOLD} mV\n")
        f.write(f"# Total spikes: {len(results['spike_times'])}\n")
        f.write(f"# Timestep (dt={results['dt']} ms)\n")
        f.write("# Format: timestep_index time_ms\n")
        for t in results['spike_times']:
            f.write(f"{t} {t * results['dt']:.3f}\n")
    print(f"\nOutput spikes saved to {output_file}")
    
    # Visualize results
    print("\nGenerating visualization...")
    plot_neuron_response(results, spike_train[:timesteps_to_simulate])
    
    print("\n" + "="*60)
    print("SIMULATION COMPLETE")
    print("="*60)

if __name__ == "__main__":
    main()
