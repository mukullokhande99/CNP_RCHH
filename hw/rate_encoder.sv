// =============================================================================
// rate_encoder.sv - Verilator Compatible
// =============================================================================
// This module implements a synthesizable rate encoder for a 28x28 image.
// It takes a static image loaded into its memory and, upon a start signal,
// streams out a time-varying spike train. The probability of a spike for a
// given pixel is proportional to its 8-bit intensity.
//
// Refactored to be compatible with Verilator and follow best practices.

`timescale 1ns / 1ps

module rate_encoder #(
    parameter int IMAGE_WIDTH   = 28,
    parameter int IMAGE_HEIGHT  = 28,
    parameter int NUM_STEPS     = 350, // Number of time steps to encode over
    parameter int PIXEL_BITS    = 8
)(
    input  logic clk,
    input  logic rst_n,

    // Control Signals
    input  logic start_encoding, // Pulse to start the encoding process

    // Output Signals
    output logic [IMAGE_WIDTH*IMAGE_HEIGHT-1:0] spike_vector,
    output logic data_valid,
    output logic encoding_busy
);
    localparam int NUM_PIXELS = IMAGE_WIDTH * IMAGE_HEIGHT;
    localparam int COUNTER_WIDTH = $clog2(NUM_STEPS);
    localparam logic [COUNTER_WIDTH-1:0] LAST_STEP = COUNTER_WIDTH'(NUM_STEPS - 1);

    // --- Image Memory ---
    // This BRAM is intended to be loaded by a testbench.
    // The pragma below makes it accessible from C++ and suppresses UNDRIVEN warnings.
    logic [PIXEL_BITS-1:0] image_mem [0:NUM_PIXELS-1] /* verilator public_flat_rw */;

    // --- State Machine ---
    typedef enum logic [1:0] {IDLE, ENCODE, DONE} state_t;
    state_t current_state, next_state;

    // --- Registers ---
    logic [COUNTER_WIDTH-1:0] step_counter;
    logic [PIXEL_BITS-1:0] lfsr_reg;

    // --- Combinational Signals ---
    logic encoding_finished;
    logic [NUM_PIXELS-1:0] generated_spikes;
    logic lfsr_next_bit;

    // --- LFSR Logic (Pseudo-Random Number Generator) ---
    assign lfsr_next_bit = lfsr_reg[7] ^ lfsr_reg[5] ^ lfsr_reg[4] ^ lfsr_reg[3];

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            lfsr_reg <= 8'hA8; // Non-zero seed
        end else if (encoding_busy) begin
            lfsr_reg <= {lfsr_reg[6:0], lfsr_next_bit};
        end
    end

    // --- Spike Generation Logic ---
    genvar i;
    generate
        for (i = 0; i < NUM_PIXELS; i = i + 1) begin : spike_gen_loop
            assign generated_spikes[i] = (lfsr_reg < image_mem[i]);
        end
    endgenerate

    // --- FSM Sequential Logic ---
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) current_state <= IDLE;
        else        current_state <= next_state;
    end

    // --- FSM Combinational Logic ---
    always_comb begin
        next_state = current_state;
        case (current_state)
            IDLE:   if (start_encoding) next_state = ENCODE;
            ENCODE: if (encoding_finished) next_state = DONE;
            DONE:   next_state = IDLE;
            default: next_state = IDLE;
        endcase
    end

    // --- Counter Logic ---
    assign encoding_finished = (step_counter == LAST_STEP);

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            step_counter <= '0;
        end else if (current_state != ENCODE) begin
            // Reset counter if not actively encoding
            step_counter <= '0;
        end else begin // current_state == ENCODE
            step_counter <= step_counter + 1;
        end
    end

    // --- Output Logic ---
    assign encoding_busy = (current_state == ENCODE);
    assign data_valid    = (current_state == ENCODE);
    assign spike_vector  = (current_state == ENCODE) ? generated_spikes : '0;

endmodule
