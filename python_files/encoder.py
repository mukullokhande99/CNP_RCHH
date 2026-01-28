import numpy as np
from torchvision import datasets, transforms

def generate_cifar_hex(image_index=0, timesteps=350, output_file="cifar_input.hex"):
    # 1. Load CIFAR-10 (Grayscale for 1024 inputs)
    transform = transforms.Compose([transforms.Grayscale(), transforms.ToTensor()])
    dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    img, _ = dataset[image_index]
    pixels = img.view(-1).numpy()
    
    # 2. Rate Coding
    spike_train = (np.random.rand(timesteps, 1024) < pixels).astype(int)
    
    # 3. Save as Hex (1024 bits per line = 256 hex digits)
    with open(output_file, 'w') as f:
        for t in range(timesteps):
            binary_str = "".join(map(str, spike_train[t]))
            hex_val = hex(int(binary_str, 2))[2:].zfill(256)
            f.write(f"{hex_val}\n")
    print(f"Generated {output_file} ({timesteps} timesteps)")

if __name__ == "__main__":
    generate_cifar_hex()
