import numpy as np
import time
import csv
import os
import matplotlib.pyplot as plt
from neuron import hh_step, to_fixed_point, to_float, SCALE
from network import SNNLayer

# ==============================================================================
# RIGOROUS METRIC DEFINITION (For Reviewer Rebuttal)
# ==============================================================================
# Refined calculation based on CORDIC-HH complexity:
# - Gating (alpha/beta): 12 CORDIC-EXP, 12 DIV, 24 ADD/SUB
# - Conductance/ODE: ~130 fixed-point operations
# Total estimate: ~180 operations per neuron per step.
OPS_PER_HH_NEURON_STEP = 180 

def run_scaling_analysis(hex_file, resolutions=[28, 32, 64, 128, 224], widths=[256, 512, 1024], csv_file="scaling_results.csv"):
    """
    Performs a systematic scaling investigation to address Reviewer concerns.
    Sweeps from 28x28 (MNIST) up to 224x224 (ImageNet Scale).
    """
    dt_val = 0.0625
    timesteps = 350 # Biological time window (21.8 ms)
    
    # Updated fieldnames to explicitly include MSOPs and GOPS
    fieldnames = ["timestamp", "resolution", "hidden_width", "total_neurons", "total_synapses", 
                  "execution_time_sec", "latency_per_step_ms", "MSOPs", "GOPS", "normalized_latency"]
    
    file_exists = os.path.isfile(csv_file)
    with open(csv_file, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists: writer.writeheader()

        for res in resolutions:
            for w in widths:
                input_dim = res * res
                dims = [input_dim, w, 10]
                print(f"\n[SCALING] Testing Resolution {res}x{res} ({input_dim} inputs) | Hidden Width {w}")
                
                # 1. Initialize Network with scaled parameters
                layers = []
                for i in range(len(dims)-1):
                    # Maintain signal scaling as dimensionality increases
                    layer_weight_loc = 15.0 + (i * 2.0)
                    layers.append(SNNLayer(dims[i], dims[i+1], weight_loc=layer_weight_loc, weight_scale=2.0, dt=dt_val))
                
                # 2. Simulate & Profile (Using mock input to stress-test high resolution)
                mock_spikes = np.random.randint(0, 2, (timesteps, input_dim))
                
                start_time = time.perf_counter()
                current_input = mock_spikes
                for layer in layers:
                    layer.reset()
                    layer_out = []
                    for t in range(timesteps):
                        # Synaptic Current Calculation (Dot product phase)
                        I_inj = layer.compute_current(current_input[t])
                        # Neuron Dynamics Phase (CORDIC-HH ODE solving)
                        spks = layer.step(I_inj, t * dt_val)
                        layer_out.append(spks)
                    current_input = np.array(layer_out)
                end_time = time.perf_counter()

                # 3. Metrics Calculation
                total_time = end_time - start_time
                total_neurons = sum(dims[1:])
                total_synapses = sum(dims[i] * dims[i+1] for i in range(len(dims)-1))
                
                # MSOPs: Focuses on connectivity (Synaptic Operations)
                # Formula: (Synapses * Timesteps) / Time / 10^6
                msops = (total_synapses * timesteps / total_time) / 1e6
                
                # GOPS: Focuses on total computational load (Synapses + HH Math)
                # Formula: ((Synapses + (Neurons * 180 ops)) * Timesteps) / Time / 10^9
                total_ops = (total_synapses * timesteps) + (total_neurons * timesteps * OPS_PER_HH_NEURON_STEP)
                gops = (total_ops / total_time) / 1e9
                
                # Latency metrics
                lat_step = (total_time / timesteps) * 1000
                norm_lat = total_time / (res * res) # Seconds per input pixel

                result = {
                    "timestamp": time.strftime("%H:%M:%S"),
                    "resolution": f"{res}x{res}",
                    "hidden_width": w,
                    "total_neurons": total_neurons,
                    "total_synapses": total_synapses,
                    "execution_time_sec": round(total_time, 4),
                    "latency_per_step_ms": round(lat_step, 4),
                    "MSOPs": round(msops, 2),
                    "GOPS": round(gops, 4),
                    "normalized_latency": round(norm_lat, 6)
                }
                writer.writerow(result)
                print(f" >> Results for {res}x{res}: {msops:.2f} MSOPs | {gops:.4f} GOPS | {norm_lat:.6f} s/px")

if __name__ == "__main__":
    # This run will take longer but will satisfy the reviewer's request for 224x224 scaling
    run_scaling_analysis("cifar_input.hex")
