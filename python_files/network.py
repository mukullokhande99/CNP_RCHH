#!/usr/bin/env python3
"""
Multi-layer Spiking Neural Network
====================================
Implements a dense, feed-forward SNN architecture.
- Neurons in each layer are Hodgkin-Huxley models from neuron.py.
- Takes a spike train from rate-encoded MNIST data.
- Propagates spikes layer-by-layer.
- The final prediction is the output neuron with the highest spike count.
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

from neuron import hh_step, to_fixed_point, to_float, SPIKE_THRESHOLD, REFRACTORY_PERIOD

# ==============================================================================
# SNN LAYER CLASS
# ==============================================================================
class SNNLayer:
    """A single layer of Hodgkin-Huxley neurons."""
    def __init__(self, n_inputs, n_neurons, weight_loc, weight_scale, dt=0.0625, is_input_layer=False):
        self.n_inputs = n_inputs
        self.n_neurons = n_neurons
        self.dt = dt
        self.is_input_layer = is_input_layer

        if is_input_layer:
            w_shape = (n_neurons, 4)
        else:
            w_shape = (n_inputs, n_neurons)
        
        self.initial_weights_fp = to_fixed_point(
            np.random.normal(loc=weight_loc, scale=weight_scale, size=w_shape)
        )
        self.reset()

    def reset(self):
        """Resets all neuron states to their initial resting values."""
        self.V_fp = np.full(self.n_neurons, to_fixed_point(-65.0), dtype=np.int64)
        self.m_fp = np.full(self.n_neurons, to_fixed_point(0.05), dtype=np.int64)
        self.h_fp = np.full(self.n_neurons, to_fixed_point(0.6), dtype=np.int64)
        self.n_fp = np.full(self.n_neurons, to_fixed_point(0.32), dtype=np.int64)
        self.V_prev_fp = np.copy(self.V_fp)
        self.last_spike_time = np.full(self.n_neurons, -np.inf)
        self.weights_fp = np.copy(self.initial_weights_fp)

    def compute_current(self, input_spikes):
        if self.is_input_layer:
            reshaped_spikes = input_spikes.reshape(self.n_neurons, 4)
            I_inj_fp = np.sum(reshaped_spikes * self.weights_fp, axis=1)
        else:
            I_inj_fp = np.dot(input_spikes, self.weights_fp)
        return I_inj_fp.astype(np.int64)

    def step(self, I_inj_fp, current_time_ms):
        output_spikes = np.zeros(self.n_neurons, dtype=int)
        self.V_prev_fp = np.copy(self.V_fp)

        for i in range(self.n_neurons):
            self.V_fp[i], self.m_fp[i], self.h_fp[i], self.n_fp[i] = hh_step(
                self.V_fp[i], self.m_fp[i], self.h_fp[i], self.n_fp[i],
                I_inj_fp[i], self.dt
            )

        V_current = to_float(self.V_fp)
        V_previous = to_float(self.V_prev_fp)
        spiked_cond1 = (V_previous <= SPIKE_THRESHOLD) & (V_current > SPIKE_THRESHOLD)
        refractory_cond2 = (current_time_ms - self.last_spike_time) > REFRACTORY_PERIOD
        spiked_neurons = np.where(spiked_cond1 & refractory_cond2)[0]
        
        if spiked_neurons.size > 0:
            output_spikes[spiked_neurons] = 1
            self.last_spike_time[spiked_neurons] = current_time_ms
        return output_spikes

# ==============================================================================
# SNN NETWORK CLASS
# ==============================================================================
class SNNNetwork:
    """Manages the full SNN, including all layers and the simulation process."""
    def __init__(self, network_config, input_size=784, dt=0.0625):
        self.network_config = network_config
        self.input_size = input_size
        self.dt = dt
        self.layers = []
        
        print("Initializing SNN with layer-specific weights...")
        
        n_in = self.input_size
        for i, layer_config in enumerate(self.network_config):
            n_out, loc, scale = layer_config['neurons'], layer_config['loc'], layer_config['scale']
            is_input = (i == 0)
            
            self.layers.append(SNNLayer(
                n_inputs=n_in, n_neurons=n_out, weight_loc=loc, 
                weight_scale=scale, dt=dt, is_input_layer=is_input
            ))
            print(f"  - Layer {i+1}: {n_in} inputs -> {n_out} neurons (w_mean={loc})")
            n_in = n_out

    def reset(self):
        """Resets all layers in the network to their initial states."""
        print("Resetting network states...")
        for layer in self.layers:
            layer.reset()

    def run(self, spike_train, timesteps_to_simulate):
        self.reset()
        num_output_neurons = self.layers[-1].n_neurons
        output_spike_recorder = np.zeros((timesteps_to_simulate, num_output_neurons), dtype=int)
        
        print(f"\nRunning simulation for {timesteps_to_simulate} timesteps...")
        
        for t in tqdm(range(timesteps_to_simulate)):
            current_time = t * self.dt
            spikes_into_layer = spike_train[t]
            
            # --- Enhanced Debug Block Start ---
            if t % 50 == 0:
                print(f"\n--- Timestep t={t} ---")
                print(f"Input image delivered {np.sum(spikes_into_layer)} spikes to Layer 1")
            # --- Enhanced Debug Block End ---

            for i, layer in enumerate(self.layers):
                I_inj_fp = layer.compute_current(spikes_into_layer)
                spikes_out_of_layer = layer.step(I_inj_fp, current_time)

                # --- Enhanced Debug Block Start ---
                if t % 50 == 0:
                    max_I_inj = to_float(np.max(I_inj_fp)) if np.any(I_inj_fp) else 0.0
                    num_output_spikes = np.sum(spikes_out_of_layer)
                    print(f"  Layer {i+1}: I_inj max: {max_I_inj:7.2f} µA/cm²  ->  Produced {num_output_spikes} spikes")
                # --- Enhanced Debug Block End ---

                spikes_into_layer = spikes_out_of_layer

            output_spike_recorder[t] = spikes_into_layer
        
        print("\nSimulation complete.")
        return output_spike_recorder

    def predict(self, output_spikes):
        """Determines the predicted class based on which output neuron spiked the most."""
        total_spikes_per_neuron = np.sum(output_spikes, axis=0)
        
        print("\n--- Prediction Results ---")
        for i, count in enumerate(total_spikes_per_neuron):
            print(f"  Neuron {i}: {count} spikes")
        
        if np.sum(total_spikes_per_neuron) == 0:
            print("--------------------------")
            print("No output spikes detected. No prediction.")
            return -1
        
        # Find all neurons that have the maximum spike count
        max_spikes = np.max(total_spikes_per_neuron)
        best_neurons = np.where(total_spikes_per_neuron == max_spikes)[0]
        prediction = best_neurons[0] # In case of a tie, pick the first one

        print(f"--------------------------")
        if len(best_neurons) > 1:
            print(f"Predicted Digit: {prediction} (Tie between neurons: {best_neurons})")
        else:
            print(f"Predicted Digit: {prediction}")
        return prediction

# ==============================================================================
# VISUALIZATION
# ==============================================================================
def plot_output_spikes(spike_data, dt, save_path='snn_output_spikes.png'):
    fig, ax = plt.subplots(figsize=(12, 6))
    
    spike_counts_exist = False
    for neuron_idx in range(spike_data.shape[1]):
        spike_times = np.where(spike_data[:, neuron_idx])[0] * dt
        if len(spike_times) > 0:
            ax.vlines(spike_times, neuron_idx - 0.4, neuron_idx + 0.4)
            spike_counts_exist = True

    if not spike_counts_exist:
        ax.text(0.5, 0.5, 'No Output Spikes', ha='center', va='center',
                transform=ax.transAxes, fontsize=15, color='gray')

    ax.set_yticks(range(spike_data.shape[1]))
    ax.set_yticklabels([f"Neuron {i}" for i in range(spike_data.shape[1])])
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Output Neuron')
    ax.set_title('Output Layer Spike Raster Plot')
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax.set_ylim(-0.5, spike_data.shape[1] - 0.5)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    # plt.show()

# ==============================================================================
# MAIN EXECUTION
# ==============================================================================
def main():
    """Main function to build and run the SNN."""
    print("="*60)
    print("SNN ARCHITECTURE SIMULATION")
    print("="*60)

    spike_file = 'spike_train_verilog.txt'
    if not os.path.exists(spike_file):
        print(f"ERROR: {spike_file} not found.")
        return
    
    print(f"Loading spike train from {spike_file}...")
    with open(spike_file, 'r') as f:
        lines = f.readlines()
    spike_train = np.array([list(map(int, line.strip())) for line in lines], dtype=int)
    print(f"Spike train shape: {spike_train.shape}")
    
    timesteps_to_simulate = 350
    dt_val = 0.0625

    # Tuned weights to ensure signal propagation for the baseline network.
    network_config = [
        {'neurons': 196, 'loc': 3.5,  'scale': 0.5},
        {'neurons': 64,  'loc': 15.0, 'scale': 3.0},
        {'neurons': 32,  'loc': 20.0, 'scale': 4.0},
        {'neurons': 32,  'loc': 25.0, 'scale': 4.0},
        {'neurons': 10,  'loc': 30.0, 'scale': 5.0},
    ]
    
    network = SNNNetwork(network_config=network_config, dt=dt_val)

    # Run Simulation (with fixed weights, no STDP)
    final_layer_spikes = network.run(
        spike_train,
        timesteps_to_simulate=min(timesteps_to_simulate, spike_train.shape[0])
    )
    
    network.predict(final_layer_spikes)
    
    print("\nGenerating visualization...")
    plot_output_spikes(final_layer_spikes, dt=dt_val)

if __name__ == "__main__":
    main()