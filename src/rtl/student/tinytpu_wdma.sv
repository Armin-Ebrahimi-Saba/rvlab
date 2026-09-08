// SPDX-License-Identifier: SHL-2.1
// SPDX-FileCopyrightText: 2024 RVLab Contributors

// tiny-tpu weight DMA: fills a data buffer straight from the main crossbar.
//
// Without this, every weight byte reaches the array through a CPU load/store
// pair. One ViT-S block holds ~1.8 MB of int8 weights, which is both far more
// than fits in the on-chip buffers and far more than the core should be
// touching by hand. The DMA turns "load the next weight tile" into three
// register writes and a poll.
//
// It reads 32-bit words because that is what TL-UL carries, and assembles four
// of them into the 128-bit word the buffers store. The destination is a word
// index inside one of the fast-port regions, not an address, so the descriptor
// says nothing about the SoC memory map on the write side.
//
// One request is outstanding at a time. tlul_adapter_host drives a_source to a
// constant zero, and TL-UL only permits one in-flight request per source id, so
// pipelining would need a source counter and a reorder buffer here. That is the
// obvious next optimisation -- at 50 MHz, one word per round trip is roughly
// 4 bytes per bus round trip rather than per cycle -- but it changes throughput,
// not behaviour, and correctness comes first.
//
// Nothing arbitrates between the DMA and the array on the buffer write port:
// software runs the DMA while the engine is idle, exactly as it fills the
// buffers by hand today. Starting a DMA into a buffer the array is reading is a
// driver bug, and status.busy is there to make that checkable.

module tinytpu_wdma #(
  parameter int WORD_BITS = 128,      // buffer word width
  parameter int DST_AW    = 16        // destination word-index width
) (
  input  logic clk_i,
  input  logic rst_ni,

  // descriptor
  input  logic               start_i,
  input  logic [31:0]        src_addr_i,
  input  logic [DST_AW-1:0]  dst_word_i,
  input  logic [1:0]         dst_region_i,
  input  logic [16:0]        len_i,        // in WORD_BITS-wide words

  output logic               busy_o,
  output logic               done_o,       // single-cycle pulse
  output logic               err_o,        // single-cycle pulse, TL error response

  // buffer write port
  output logic                 wr_en_o,
  output logic [1:0]           wr_region_o,
  output logic [DST_AW-1:0]    wr_addr_o,
  output logic [WORD_BITS-1:0] wr_data_o,

  // host side of the main crossbar
  output tlul_pkg::tl_h2d_t tl_o,
  input  tlul_pkg::tl_d2h_t tl_i
);

  localparam int LANES = WORD_BITS / 32;
  localparam int LB    = $clog2(LANES);

  typedef enum logic [2:0] { S_IDLE, S_REQ, S_RSP, S_WR, S_DONE } state_e;
  state_e state;

  logic [31:0]        src_q;
  logic [DST_AW-1:0]  dst_q;
  logic [1:0]         region_q;
  logic [16:0]        rem_q;
  logic [LB-1:0]      lane_q;
  logic [WORD_BITS-1:0] sh_q;
  logic               err_q;

  // ------------------------------------------------------------- TL-UL host
  logic        req, gnt, rsp_valid;
  logic [31:0] rsp_data;

  tlul_adapter_host #(.AW(32), .DW(32)) u_host (
    .clk_i,
    .rst_ni,
    .req_i   (req),
    .gnt_o   (gnt),
    .addr_i  (src_q),
    .we_i    (1'b0),
    .wdata_i ('0),
    .be_i    (4'hf),
    .size_i  (2'd2),          // 2**2 = 4 bytes
    .valid_o (rsp_valid),
    .rdata_o (rsp_data),
    .tl_o,
    .tl_i
  );

  assign req    = (state == S_REQ);
  assign busy_o = (state != S_IDLE);

  // ------------------------------------------------------------------- FSM
  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      state    <= S_IDLE;
      src_q    <= '0;
      dst_q    <= '0;
      region_q <= '0;
      rem_q    <= '0;
      lane_q   <= '0;
      sh_q     <= '0;
      err_q    <= 1'b0;
      wr_en_o     <= 1'b0;
      wr_region_o <= '0;
      wr_addr_o   <= '0;
      wr_data_o   <= '0;
      done_o      <= 1'b0;
      err_o       <= 1'b0;
    end else begin
      wr_en_o <= 1'b0;
      done_o  <= 1'b0;
      err_o   <= 1'b0;

      unique case (state)
        S_IDLE: begin
          if (start_i && len_i != '0) begin
            // The low four bits are dropped rather than honoured: a 128-bit
            // word straddling two source words would need a barrel shifter on
            // the read path for no benefit, since every producer of a weight
            // tile aligns it.
            src_q    <= {src_addr_i[31:4], 4'b0000};
            dst_q    <= dst_word_i;
            region_q <= dst_region_i;
            rem_q    <= len_i;
            lane_q   <= '0;
            err_q    <= 1'b0;
            state    <= S_REQ;
          end
        end

        S_REQ: begin
          if (gnt) state <= S_RSP;
        end

        S_RSP: begin
          if (rsp_valid) begin
            sh_q[32*lane_q +: 32] <= rsp_data;
            // An error response still advances the transfer. Stalling would
            // leave busy asserted forever and hang the poll; the sticky error
            // flag is how the driver finds out.
            if (tl_i.d_error) err_q <= 1'b1;
            src_q <= src_q + 32'd4;
            if (lane_q == LB'(LANES - 1)) begin
              lane_q <= '0;
              state  <= S_WR;
            end else begin
              lane_q <= lane_q + LB'(1);
              state  <= S_REQ;
            end
          end
        end

        S_WR: begin
          // sh_q is complete only now: the last lane was written by the
          // non-blocking assignment that ended S_RSP.
          wr_en_o     <= 1'b1;
          wr_region_o <= region_q;
          wr_addr_o   <= dst_q;
          wr_data_o   <= sh_q;
          dst_q       <= dst_q + DST_AW'(1);
          rem_q       <= rem_q - 17'd1;
          state       <= (rem_q == 17'd1) ? S_DONE : S_REQ;
        end

        S_DONE: begin
          done_o <= 1'b1;
          err_o  <= err_q;
          state  <= S_IDLE;
        end

        default: state <= S_IDLE;
      endcase
    end
  end

endmodule
