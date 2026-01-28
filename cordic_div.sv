// =============================================================================
// cordic_div.sv - CORDIC-based Division Unit (Verilator Compatible)
// =============================================================================
// Implements CORDIC division using vectoring mode.
// Uses types and parameters from the shared fixed-point package.
// Refactored for clean separation of combinational and sequential logic.

`timescale 1ns / 1ps
`include "fixed_point_defs.svh"

module cordic_div (
    input  logic clk,
    input  logic rst_n,
    input  logic start,
    output logic done,
    input  fixed_point_t x_in,        // Dividend
    input  fixed_point_t y_in,        // Divisor
    output fixed_point_t z_out        // Quotient
);
    import fixed_point_pkg::*;

    // FSM states
    enum logic [1:0] { IDLE, CALCULATE, COMPLETE } state, next_state;

    // Registers (Sequential)
    fixed_point_div_t x_reg;
    fixed_point_div_t z_reg;
    fixed_point_div_t y_reg;
    logic signed [4:0] iter_counter; // Signed to handle -k to +k
    logic result_sign;

    // Combinational Wires
    logic iter_done;
    logic divide_by_zero;
    fixed_point_div_t y_scaled;
    fixed_point_div_t term_2_pow_neg_i;
    fixed_point_t final_quotient;

    // FSM State Register
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) state <= IDLE;
        else        state <= next_state;
    end

    // FSM Next State & Output Logic
    always_comb begin
        next_state = state;
        done = 1'b0;
        case (state)
            IDLE:       if (start) next_state = CALCULATE;
            CALCULATE:  if (iter_done) next_state = COMPLETE;
            COMPLETE:   begin done = 1'b1; if (!start) next_state = IDLE; end
            default:    next_state = IDLE;
        endcase
    end

    // Combinational Datapath
    always_comb begin
        // Iteration completion check
        iter_done = (iter_counter > DIV_ITER);
        divide_by_zero = (y_in == FP_ZERO);

        // CORDIC shifter logic
        if (iter_counter >= 0) begin
            term_2_pow_neg_i = extend_to_div(FP_ONE) >>> iter_counter;
            y_scaled         = y_reg >>> iter_counter;
        end else begin
            term_2_pow_neg_i = extend_to_div(FP_ONE) <<< (-iter_counter);
            y_scaled         = y_reg <<< (-iter_counter);
        end

        // Final output logic with sign correction and zero handling
        final_quotient = z_reg[MAIN_WIDTH-1:0];
        if (divide_by_zero) begin
            z_out = (x_in[MAIN_WIDTH-1] ? FP_MAX_NEG : FP_MAX_POS);
        end else begin
            z_out = (result_sign ? -final_quotient : final_quotient);
        end
    end

    // Sequential Datapath
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            x_reg <= '0;
            z_reg <= '0;
            y_reg <= '0;
            iter_counter <= -DIV_ITER;
            result_sign <= 1'b0;
        end else begin
            case (state)
                IDLE: begin
                    if (start) begin
                        x_reg <= extend_to_div(fp_abs(x_in));
                        y_reg <= extend_to_div(fp_abs(y_in));
                        z_reg <= '0;
                        iter_counter <= -DIV_ITER;
                        result_sign <= x_in[MAIN_WIDTH-1] ^ y_in[MAIN_WIDTH-1];
                    end
                end
                CALCULATE: begin
                    if (!iter_done) begin
                        if (x_reg >= 0) begin
                            x_reg <= x_reg - y_scaled;
                            z_reg <= z_reg + term_2_pow_neg_i;
                        end else begin
                            x_reg <= x_reg + y_scaled;
                            z_reg <= z_reg - term_2_pow_neg_i;
                        end
                        iter_counter <= iter_counter + 1;
                    end
                end
                default: ; // Hold register values
            endcase
        end
    end

endmodule

