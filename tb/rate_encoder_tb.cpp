#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <iomanip>
#include "Vrate_encoder.h"      // Verilator generated header for the DUT
#include "Vrate_encoder___024root.h" // Include internal header for signal access
#include "verilated.h"
#include "verilated_vcd_c.h"

// C++ constants for module parameters
constexpr int IMAGE_WIDTH = 28;
constexpr int IMAGE_HEIGHT = 28;
constexpr int NUM_PIXELS = IMAGE_WIDTH * IMAGE_HEIGHT;
constexpr int NUM_STEPS = 350;

// Helper function to drive the clock for one half-period
void tick(Vrate_encoder* dut, VerilatedVcdC* tracer, vluint64_t& sim_time) {
    dut->clk = !dut->clk;
    dut->eval();
    if (tracer) tracer->dump(sim_time++);
}

// Function to read a hex file into a vector of bytes, IGNORING comments
// This mimics SystemVerilog's $readmemh more closely.
bool read_hex_file(const std::string& filename, std::vector<uint8_t>& memory) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "ERROR: Could not open hex file: " << filename << std::endl;
        return false;
    }

    memory.clear();
    std::string line;
    // C++ FIX: Use std::getline to read the whole line to properly handle comments
    while (std::getline(file, line)) {
        // Trim leading whitespace (optional but good practice)
        line.erase(0, line.find_first_not_of(" \t\n\r"));
        // Trim trailing whitespace
        line.erase(line.find_last_not_of(" \t\n\r") + 1);

        // Ignore empty lines and lines that start with a comment
        if (line.empty() || line.rfind("//", 0) == 0) {
            continue;
        }

        try {
            memory.push_back(static_cast<uint8_t>(std::stoul(line, nullptr, 16)));
        } catch (const std::exception& e) {
            std::cerr << "ERROR: Invalid hex value '" << line << "' in file " << filename << std::endl;
            return false;
        }
    }
    return true;
}

// Main Simulation Logic
int main(int argc, char** argv) {
    // --- Initialization ---
    Verilated::commandArgs(argc, argv);
    Vrate_encoder* dut = new Vrate_encoder;
    Verilated::traceEverOn(true);
    VerilatedVcdC* tfp = new VerilatedVcdC;
    dut->trace(tfp, 99);
    tfp->open("rate_encoder.vcd");
    vluint64_t main_time = 0;

    std::cout << "--- Starting C++ Testbench for Rate Encoder ---" << std::endl;

    // --- File I/O Handles ---
    const std::string image_filename = "mnist_image.hex";
    const std::string spike_filename = "spike_train_verilog.txt";
    std::ofstream spike_file(spike_filename);

    if (!spike_file.is_open()) {
        std::cerr << "ERROR: Could not open " << spike_filename << " for writing." << std::endl;
        return 1;
    }

    // --- Load Image Data into DUT Memory ---
    std::vector<uint8_t> image_data;
    if (!read_hex_file(image_filename, image_data) || image_data.size() != NUM_PIXELS) {
        if (image_data.size() != NUM_PIXELS) {
            std::cerr << "ERROR: Expected " << NUM_PIXELS << " pixels, but found " << image_data.size() << std::endl;
        }
        return 1;
    }
    // Access the public memory via the root pointer
    for (size_t i = 0; i < image_data.size(); ++i) {
        dut->rootp->rate_encoder__DOT__image_mem[i] = image_data[i];
    }
    std::cout << "Loaded " << image_filename << " into encoder memory." << std::endl;

    // --- Reset Sequence ---
    dut->clk = 0;
    dut->rst_n = 0;
    dut->start_encoding = 0;
    for (int i = 0; i < 4; ++i) tick(dut, tfp, main_time);
    dut->rst_n = 1;
    std::cout << "Reset released." << std::endl;

    // --- Start the Encoding Process ---
    tick(dut, tfp, main_time);
    std::cout << "Pulsing start_encoding..." << std::endl;
    dut->start_encoding = 1;
    tick(dut, tfp, main_time);
    tick(dut, tfp, main_time);
    dut->start_encoding = 0;

    // --- Wait for Encoding to Start and Finish ---
    int timeout = NUM_STEPS + 100;
    bool started = false;

    while (timeout > 0) {
        tick(dut, tfp, main_time); // posedge
        
        if (dut->encoding_busy) {
            if (!started) {
                std::cout << "Encoding has started. Capturing spike data..." << std::endl;
                started = true;
            }
            if (dut->data_valid) {
                std::string spike_str(NUM_PIXELS, '0');
                for (int i = 0; i < NUM_PIXELS; ++i) {
                    // Correctly reconstruct the 784-bit vector from Verilator's 32-bit words
                    if ((dut->spike_vector[i / 32] >> (i % 32)) & 1) {
                        spike_str[i] = '1';
                    }
                }
                spike_file << spike_str << std::endl;
            }
        } else if (started) {
            break; // Encoding was busy but isn't now, so it's finished.
        }

        tick(dut, tfp, main_time); // negedge
        timeout--;
    }

    if (timeout == 0) std::cerr << "ERROR: Simulation timed out!" << std::endl;
    else std::cout << "Encoding finished." << std::endl;

    // --- End Simulation ---
    for(int i=0; i<5; ++i) tick(dut, tfp, main_time);
    spike_file.close();
    tfp->close();
    delete dut;
    std::cout << "Test finished. Check rate_encoder.vcd and " << spike_filename << std::endl;
    return 0;
}
