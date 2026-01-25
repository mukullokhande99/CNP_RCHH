import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import snntorch as snn
from snntorch import spikegen
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings('ignore')

# ==============================================================================
# CONFIGURATION
# ==============================================================================
IMAGE_WIDTH = 28
IMAGE_HEIGHT = 28
NUM_STEPS = 200      # Number of timesteps
LFSR_SEED = 0xA8     # Must match the Verilog seed

# Visualization & Simulation parameters
TIME_WINDOW = 100    # Time window to display
NUM_OUTPUT_NEURONS = 10 # Number of output LIF neurons

# ==============================================================================
# DATA LOADING FUNCTION (Unchanged)
# ==============================================================================
def load_image_from_hex(filename, width, height):
    """Loads an image from a hex file."""
    print(f"Loading image data from '{filename}'...")
    try:
        # Skip comments and read hex values
        with open(filename, 'r') as f:
            lines = f.readlines()
            pixels = [int(line.strip(), 16) for line in lines if not line.strip().startswith('//')]
    except FileNotFoundError:
        print(f"Error: Hex file '{filename}' not found.")
        return None
    except ValueError:
        print(f"Error: Invalid hex value in '{filename}'.")
        return None

    if len(pixels) != width * height:
        print(f"Error: Incorrect number of pixels in '{filename}'. Expected {width*height}, got {len(pixels)}.")
        return None

    image_array = np.array(pixels, dtype=np.uint8).reshape((height, width))
    print("Image loaded successfully.")
    return image_array

# ==============================================================================
# RATE ENCODING FUNCTION (Unchanged)
# ==============================================================================
def generate_rate_encoded_spikes(image_data, num_steps):
    """Generates rate-encoded spike patterns from image data."""
    print("--- Generating Rate-Encoded Spikes ---")
    image_flat = image_data.flatten()
    lfsr_reg = LFSR_SEED
    spike_patterns = np.zeros((num_steps, len(image_flat)), dtype=bool)
    
    for t in range(num_steps):
        current_spikes = (lfsr_reg < image_flat)
        spike_patterns[t, :] = current_spikes
        
        bit7 = (lfsr_reg >> 7) & 1
        bit5 = (lfsr_reg >> 5) & 1
        bit4 = (lfsr_reg >> 4) & 1
        bit3 = (lfsr_reg >> 3) & 1
        new_lsb = bit7 ^ bit5 ^ bit4 ^ bit3
        lfsr_reg = ((lfsr_reg << 1) & 0xFF) | new_lsb
    
    print(f"Generated spike patterns for {num_steps} timesteps")
    return spike_patterns

# ==============================================================================
# LIF LAYER SIMULATION (Unchanged)
# ==============================================================================
def simulate_lif_layer(spike_inputs, num_outputs=10, beta=0.9, threshold=1.0):
    """Simulates a layer of LIF neurons."""
    print(f"--- Simulating a Layer of {num_outputs} LIF Neurons ---")
    num_timesteps, num_inputs = spike_inputs.shape
    spike_tensor = torch.tensor(spike_inputs, dtype=torch.float32)
    
    fc_layer = nn.Linear(num_inputs, num_outputs)
    lif_layer = snn.Leaky(beta=beta, threshold=threshold)
    
    mem_pot_layer = torch.zeros(num_timesteps, num_outputs)
    spk_out_layer = torch.zeros(num_timesteps, num_outputs)
    mem = lif_layer.init_leaky()
    
    for t in range(num_timesteps):
        cur_in = fc_layer(spike_tensor[t])
        spk_out, mem = lif_layer(cur_in, mem)
        mem_pot_layer[t, :] = mem
        spk_out_layer[t, :] = spk_out
        
    print(f"Simulation complete.")
    return mem_pot_layer.detach().numpy(), spk_out_layer.detach().numpy().astype(bool)

# ==============================================================================
# NEW: FUNCTION TO PLOT ONE ROW FOR A SINGLE DIGIT
# ==============================================================================
def plot_digit_row(fig, axes_row, image_data, spike_patterns, mem_potentials_layer, output_spikes_layer, digit):
    """
    Plots the three visualizations for a single digit on a given row of axes.
    """
    print(f"--- Plotting visualization for Digit: {digit} ---")
    
    ax1, ax2, ax3_main = axes_row
    time_axis = np.arange(TIME_WINDOW)

    # 1. Input Image
    ax1.imshow(image_data, cmap='gray', interpolation='nearest')
    ax1.set_title(f'Input Image (Digit {digit})', fontsize=12, fontweight='bold')
    ax1.set_xticks([])
    ax1.set_yticks([])

    # 2. Input Spike Raster Plot
    spike_times, spike_neurons = np.where(spike_patterns[:TIME_WINDOW, :])
    ax2.scatter(spike_times, spike_neurons, s=1, c='blue', alpha=0.8)
    ax2.set_title('Input Spike', fontsize=12, fontweight='bold')
    ax2.set_xlim(0, TIME_WINDOW)
    ax2.set_ylim(0, IMAGE_WIDTH * IMAGE_HEIGHT)
    if digit != 0: # Hide x labels for all but the last row
        ax2.set_xticklabels([])
    else:
        ax2.set_xlabel('Time Steps')
    ax2.set_ylabel('Neuron Index')


    # 3. Based on LIF Model (Stacked Plot)
    gs_lif = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=ax3_main.get_subplotspec(), height_ratios=[1, 2], hspace=0.1)
    
    ax3_top = fig.add_subplot(gs_lif[0]) # Output spikes
    ax3_bot = fig.add_subplot(gs_lif[1], sharex=ax3_top) # Membrane potential
    
    ax3_top.set_title('Based on HH Model', fontsize=12, fontweight='bold')
    
    # Plot output spikes (top)
    spike_times_out, spike_neurons_out = np.where(output_spikes_layer[:TIME_WINDOW, :])
    ax3_top.scatter(spike_times_out, spike_neurons_out, s=25, c='red', marker='.')
    ax3_top.set_ylabel('Neuron Index')
    ax3_top.set_ylim(-0.5, NUM_OUTPUT_NEURONS - 0.5)
    ax3_top.set_yticks(range(0, NUM_OUTPUT_NEURONS, 2))
    plt.setp(ax3_top.get_xticklabels(), visible=False)

    # Plot membrane potentials (bottom)
    colors = plt.cm.viridis(np.linspace(0, 1, NUM_OUTPUT_NEURONS))
    for i in range(NUM_OUTPUT_NEURONS):
        ax3_bot.plot(time_axis, mem_potentials_layer[:TIME_WINDOW, i], color=colors[i], linewidth=1.5)
    
    ax3_bot.axhline(y=1.0, color='red', linestyle='--', alpha=0.7, label='Threshold')
    ax3_bot.set_ylabel('Membrane Potential')
    ax3_bot.set_xlim(0, TIME_WINDOW)
    if digit != 0: # Hide x labels for all but the last row
         ax3_bot.set_xticklabels([])
    else:
        ax3_bot.set_xlabel('Time Steps')


# ==============================================================================
# MODIFIED MAIN FUNCTION
# ==============================================================================
def main():
    """Main function to run the script for multiple digits."""
    digits_to_process = [7, 3, 2, 1, 0]
    num_digits = len(digits_to_process)
    
    print("=" * 60)
    print("MULTI-DIGIT SPIKE ENCODING & LIF LAYER VISUALIZATION")
    print("=" * 60)

    # Create a single figure to hold all plots
    fig, axes = plt.subplots(
        num_digits, 3, 
        figsize=(15, 4 * num_digits), 
        gridspec_kw={'width_ratios': [1, 2, 2]}
    )

    for i, digit in enumerate(digits_to_process):
        print(f"\nProcessing Digit: {digit}...")
        hex_file = f"{digit}.hex"
        
        # 1. Load the image
        image = load_image_from_hex(hex_file, IMAGE_WIDTH, IMAGE_HEIGHT)
        if image is None:
            print(f"Skipping digit {digit} due to loading error.")
            continue
        
        # 2. Generate rate-encoded spikes
        spike_patterns = generate_rate_encoded_spikes(image, NUM_STEPS)
        
        # 3. Simulate the LIF neuron layer
        mem_potentials, output_spikes = simulate_lif_layer(
            spike_patterns, num_outputs=NUM_OUTPUT_NEURONS, beta=0.9, threshold=1.0
        )
        
        # 4. Plot the results for this digit on the corresponding row
        plot_digit_row(fig, axes[i], image, spike_patterns, mem_potentials, output_spikes, digit)

    # Final adjustments and saving
    fig.suptitle('Spike Encoding and HH Layer Response for Multiple Digits', fontsize=20, fontweight='bold', y=1.0)
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    
    output_filename = 'flow_visualization.png'
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    plt.show()

    print("\n" + "=" * 60)
    print(f"VISUALIZATION COMPLETE! Saved to '{output_filename}'")
    print("=" * 60)

if __name__ == "__main__":
    main()
