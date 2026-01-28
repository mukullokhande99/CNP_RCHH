import numpy as np
import imageio
import os

# ==============================================================================
# CONFIGURATION - Match these with your Verilog parameters
# ==============================================================================
IMAGE_WIDTH  = 28
IMAGE_HEIGHT = 28
NUM_STEPS    = 350      # Number of timesteps to encode over
PIXEL_BITS   = 8
LFSR_SEED    = 0xA8     # Must match the Verilog seed (8'hA8)
HEX_FILE     = 'mnist_image.hex'

# ==============================================================================
# DATA LOADING FUNCTION
# ==============================================================================
def load_image_from_hex(filename, width, height):
    """
    Loads an image from a hex file containing one pixel value per line.
    """
    print(f"Loading image data from '{filename}'...")
    try:
        with open(filename, 'r') as f:
            # Read each line, strip whitespace, and convert from hex to integer
            pixels = [int(line.strip(), 16) for line in f]
    except FileNotFoundError:
        print(f"Error: Hex file '{filename}' not found.")
        return None
    except ValueError:
        print(f"Error: Invalid content in '{filename}'. Ensure it contains only hex values.")
        return None

    expected_pixels = width * height
    if len(pixels) != expected_pixels:
        print(f"Error: Expected {expected_pixels} pixels, but found {len(pixels)}.")
        return None
        
    # Convert the flat list of pixels into a 2D numpy array and ensure type is uint8
    image_array = np.array(pixels, dtype=np.uint8).reshape((height, width))
    print("Image loaded successfully.")
    return image_array

# ==============================================================================
# HARDWARE EMULATION FUNCTION
# ==============================================================================
def emulate_rate_encoder(image_data, num_steps):
    """
    Emulates the logic of the rate_encoder.sv module in Python.
    """
    print("--- Starting Hardware Emulation ---")
    
    # In Verilog, this is a 1D memory array. Flattening simplifies the logic.
    image_flat = image_data.flatten()
    
    # Replicates: logic [PIXEL_BITS-1:0] lfsr_reg = LFSR_SEED;
    lfsr_reg = LFSR_SEED
    
    # Storage for the output spike train frames
    spike_train_frames = np.zeros((num_steps, IMAGE_HEIGHT, IMAGE_WIDTH), dtype=bool)

    # Main loop (emulates the state machine and step_counter)
    for t in range(num_steps):
        # --- Spike Generation Logic (fully parallel operation in hardware) ---
        # Replicates: generated_spikes[i] = (lfsr_reg < image_mem[i]);
        # NumPy performs this comparison for all pixels at once very efficiently.
        current_spikes_flat = (lfsr_reg < image_flat)
        
        # Reshape the flat spike vector back into a 2D frame for the GIF
        spike_train_frames[t, :, :] = current_spikes_flat.reshape(IMAGE_HEIGHT, IMAGE_WIDTH)
        
        # --- LFSR Update Logic ---
        # Replicates: lfsr_reg <= {lfsr_reg[6:0], lfsr_reg[7] ^ lfsr_reg[5] ^ lfsr_reg[4] ^ lfsr_reg[3]};
        # This is a left-shift register where the new LSB is a feedback tap.
        bit7 = (lfsr_reg >> 7) & 1
        bit5 = (lfsr_reg >> 5) & 1
        bit4 = (lfsr_reg >> 4) & 1
        bit3 = (lfsr_reg >> 3) & 1
        
        new_lsb = bit7 ^ bit5 ^ bit4 ^ bit3
        
        # Shift left, clear the new LSB, then OR it with the feedback bit.
        # The '& 0xFF' ensures the value remains 8-bit.
        lfsr_reg = ((lfsr_reg << 1) & 0xFF) | new_lsb

    print(f"--- Emulation complete. Generated {num_steps} spike frames. ---")
    return spike_train_frames

# ==============================================================================
# VISUALIZATION FUNCTION
# ==============================================================================
def create_spike_gif(spike_data, filename, fps=30):
    """
    Creates and saves a GIF from a sequence of spike frames.
    """
    frames = []
    for spike_frame in spike_data:
        # Convert boolean spike frame (True/False) to an image (white/black)
        image_frame = (spike_frame.astype(np.uint8) * 255)
        frames.append(image_frame)
    
    print(f"\nSaving GIF to '{filename}'...")
    imageio.mimsave(filename, frames, fps=fps)
    print("GIF saved successfully.")

# ==============================================================================
# MAIN SCRIPT
# ==============================================================================
def create_dummy_hex_file():
    """Creates a sample hex file if one doesn't exist."""
    print(f"'{HEX_FILE}' not found. Creating a dummy file for demonstration.")
    # Create a 28x28 image with a bright 14x14 square in the center
    dummy_image = np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH), dtype=np.uint8)
    dummy_image[7:21, 7:21] = 200 # Bright square
    dummy_image[12:16, :] = 80    # Faint horizontal bar
    
    with open(HEX_FILE, 'w') as f:
        for pixel_value in dummy_image.flatten():
            f.write(f'{pixel_value:02x}\n') # Write value as 2-digit hex
    print(f"Dummy file '{HEX_FILE}' created with a test pattern.")

if __name__ == "__main__":
    # Check if the hex file exists, if not, create one.
    if not os.path.exists(HEX_FILE):
        create_dummy_hex_file()
    
    # 1. Load the image from the specified hex file
    img_numpy = load_image_from_hex(HEX_FILE, IMAGE_WIDTH, IMAGE_HEIGHT)

    if img_numpy is not None:
        # 2. Run the hardware emulation
        hardware_spike_data = emulate_rate_encoder(img_numpy, num_steps=NUM_STEPS)

        # 3. Create and save the GIF visualization
        create_spike_gif(hardware_spike_data, "hardware_rate_encoding.gif")
