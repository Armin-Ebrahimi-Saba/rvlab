// SPDX-License-Identifier: SHL-2.1
// SPDX-FileCopyrightText: 2024 RVLab Contributors

// tiny-tpu DMA: moves data between the main crossbar and the data buffers.
//
// Without this, every byte reaches the array through a CPU load/store pair.
// One ViT-S block holds ~1.8 MB of int8 weights, which is both far more than
// fits in the on-chip buffers and far more than the core should be touching by
// hand. Three modes, chosen by dma_mode:
//
//   COPY       crossbar -> buffer, `len` contiguous 128-bit words. The original
//              weight DMA, unchanged.
//   GATHER     crossbar -> buffer, im2col rows for a k x k convolution. The
//              source is an [H][W][C] int8 tensor, C a multiple of 16 bytes
//              (Cw buffer words per pixel). Output row i is pixel (oy, ox) of
//              the output grid, starting at (oy0, ox0) and running in raster
//              order over Wo columns; it holds kernel taps t0..t1-1 in
//              row-major order, each the Cw words of one input pixel, or zeros
//              where the tap falls in the padding. Zero taps cost no bus
//              traffic. This is sw/export_head.py's im2col(), which the
//              expected values are computed from.
//   WRITEBACK  buffer -> crossbar. `rows` rows of `cols` 128-bit words, read
//              consecutively from the buffer starting at dst_word, are written
//              to DRAM with `row_stride` bytes between rows; rows come in
//              groups of `grp` (0 = one group) with `grp_stride` bytes between
//              groups' first rows -- a tile landing inside a larger tensor,
//              and a transposed conv's pixel shuffle.
//
// It moves 32-bit words because that is what TL-UL carries, and assembles or
// splits the 128-bit buffer words. Addresses below 16 bytes are dropped: every
// producer aligns its tensors.
//
// One request is outstanding at a time. tlul_adapter_host drives a_source to a
// constant zero, and TL-UL only permits one in-flight request per source id, so
// pipelining would need a source counter and a reorder buffer here. That is the
// obvious next optimisation; it changes throughput, not behaviour.
//
// Nothing arbitrates between the DMA and the engines on the buffer ports:
// software runs the DMA while the engines are idle, and status.busy is there
// to make that checkable. A write-back borrows the result buffer's first
// engine read port the same way.

module tinytpu_wdma #(
  parameter int WORD_BITS = 128,      // buffer word width
  parameter int DST_AW    = 16        // buffer word-index width
) (
  input  logic clk_i,
  input  logic rst_ni,

  // descriptor
  input  logic               start_i,
  input  logic [1:0]         mode_i,       // 0 copy, 1 gather, 2 write-back
  input  logic [31:0]        src_addr_i,   // the crossbar side, all modes
  input  logic [DST_AW-1:0]  dst_word_i,   // the buffer side, all modes
  input  logic [1:0]         dst_region_i,
  input  logic [16:0]        len_i,        // copy: in WORD_BITS-wide words
  input  logic [15:0]        rows_i,       // gather, write-back
  input  logic [15:0]        g_h_i, g_w_i, // gather: input height, width
  input  logic [15:0]        g_cw_i,       // gather: words per pixel
  input  logic [15:0]        g_wo_i,       // gather: output width
  input  logic [7:0]         g_k_i, g_stride_i,
  input  logic [3:0]         g_pad_i,
  input  logic [5:0]         g_t0_i, g_t1_i,
  input  logic [15:0]        g_oy0_i, g_ox0_i,
  input  logic [15:0]        wb_cols_i,    // write-back: words per row
  input  logic [31:0]        wb_stride_i, wb_grp_stride_i,
  input  logic [15:0]        wb_grp_i,

  output logic               busy_o,
  output logic               done_o,       // single-cycle pulse
  output logic               err_o,        // single-cycle pulse, TL error response

  // buffer write port
  output logic                 wr_en_o,
  output logic [1:0]           wr_region_o,
  output logic [DST_AW-1:0]    wr_addr_o,
  output logic [WORD_BITS-1:0] wr_data_o,

  // result-buffer read port, write-back only; data one cycle after the address
  output logic [DST_AW-1:0]    rd_addr_o,
  input  logic [WORD_BITS-1:0] rd_data_i,

  // host side of the main crossbar
  output tlul_pkg::tl_h2d_t tl_o,
  input  tlul_pkg::tl_d2h_t tl_i
);

  localparam int LANES = WORD_BITS / 32;
  localparam int LB    = $clog2(LANES);

  localparam logic [1:0] M_COPY = 2'd0, M_GATHER = 2'd1, M_WB = 2'd2;

  typedef enum logic [3:0] {
    S_IDLE, S_REQ, S_RSP, S_WR, S_TAP, S_ADDR, S_ZERO, S_RD, S_RDW, S_WREQ, S_WRSP, S_DONE
  } state_e;
  state_e state;

  logic [1:0]           mode_q;
  logic [31:0]          src_q;        // the next crossbar address
  logic [DST_AW-1:0]    dst_q;        // the next buffer word
  logic [1:0]           region_q;
  logic [16:0]          rem_q;        // copy: words left; gather: words left in this tap
  logic [LB-1:0]        lane_q;
  logic [WORD_BITS-1:0] sh_q;
  logic                 err_q;

  // gather
  logic [31:0]  base_q;
  logic [15:0]  h_q, w_q, cw_q, wo_q, oy_q, ox_q, rows_q;
  logic [7:0]   k_q, stride_q;
  logic [3:0]   pad_q;
  logic [5:0]   t_q, t0_q, t1_q;
  logic [7:0]   ky_q, kx_q, ky0_q, kx0_q;
  logic signed [17:0] iy_q, ix_q;

  // write-back
  logic [15:0]  cols_q, col_q, grp_q, gi_q;
  logic [31:0]  rstride_q, gstride_q, rowbase_q, grpbase_q;

  // ------------------------------------------------------------- TL-UL host
  logic        req, gnt, rsp_valid, we;
  logic [31:0] rsp_data, wdata;

  tlul_adapter_host #(.AW(32), .DW(32)) u_host (
    .clk_i,
    .rst_ni,
    .req_i   (req),
    .gnt_o   (gnt),
    .addr_i  (src_q),
    .we_i    (we),
    .wdata_i (wdata),
    .be_i    (4'hf),
    .size_i  (2'd2),          // 2**2 = 4 bytes
    .valid_o (rsp_valid),
    .rdata_o (rsp_data),
    .tl_o,
    .tl_i
  );

  assign req    = (state == S_REQ) || (state == S_WREQ);
  assign we     = (state == S_WREQ);
  assign wdata  = sh_q[32*lane_q +: 32];
  assign busy_o = (state != S_IDLE);
  assign rd_addr_o = dst_q;

  // Gather: where the current tap reads from. iy/ix are registered in S_TAP;
  // the bounds test and the address follow in S_ADDR.
  wire tap_in = (iy_q >= 0) && (ix_q >= 0) &&
                (iy_q < $signed({2'b0, h_q})) && (ix_q < $signed({2'b0, w_q}));
  wire [31:0] pix = 32'(iy_q[15:0]) * 32'(w_q) + 32'(ix_q[15:0]);
  wire [31:0] tap_addr = base_q + ((pix * 32'(cw_q)) << 4);

  // Advance to the next tap, or the next output pixel, or finish.
  task automatic next_tap();
    if (t_q + 6'd1 == t1_q) begin
      t_q  <= t0_q;
      ky_q <= ky0_q;
      kx_q <= kx0_q;
      if (ox_q + 16'd1 == wo_q) begin
        ox_q <= '0;
        oy_q <= oy_q + 16'd1;
      end else begin
        ox_q <= ox_q + 16'd1;
      end
      rows_q <= rows_q - 16'd1;
      state  <= (rows_q == 16'd1) ? S_DONE : S_TAP;
    end else begin
      t_q <= t_q + 6'd1;
      if (kx_q + 8'd1 == k_q) begin
        kx_q <= '0;
        ky_q <= ky_q + 8'd1;
      end else begin
        kx_q <= kx_q + 8'd1;
      end
      state <= S_TAP;
    end
  endtask

  // ------------------------------------------------------------------- FSM
  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      state    <= S_IDLE;
      mode_q   <= '0;
      src_q    <= '0;
      dst_q    <= '0;
      region_q <= '0;
      rem_q    <= '0;
      lane_q   <= '0;
      sh_q     <= '0;
      err_q    <= 1'b0;
      base_q   <= '0;
      {h_q, w_q, cw_q, wo_q, oy_q, ox_q, rows_q} <= '0;
      {k_q, stride_q, pad_q, t_q, t0_q, t1_q} <= '0;
      {ky_q, kx_q, ky0_q, kx0_q} <= '0;
      iy_q <= '0;
      ix_q <= '0;
      {cols_q, col_q, grp_q, gi_q} <= '0;
      {rstride_q, gstride_q, rowbase_q, grpbase_q} <= '0;
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
          if (start_i) begin
            mode_q   <= mode_i;
            dst_q    <= dst_word_i;
            region_q <= dst_region_i;
            lane_q   <= '0;
            err_q    <= 1'b0;
            unique case (mode_i)
              M_GATHER: begin
                base_q   <= {src_addr_i[31:4], 4'b0000};
                h_q      <= g_h_i;
                w_q      <= g_w_i;
                cw_q     <= g_cw_i;
                wo_q     <= g_wo_i;
                k_q      <= g_k_i;
                stride_q <= g_stride_i;
                pad_q    <= g_pad_i;
                t0_q     <= g_t0_i;
                t1_q     <= g_t1_i;
                t_q      <= g_t0_i;
                // t0 is at most 63 and k at most 7: a small divider, and
                // only ever on the start cycle.
                ky0_q    <= 8'(g_t0_i / g_k_i);
                kx0_q    <= 8'(g_t0_i % g_k_i);
                ky_q     <= 8'(g_t0_i / g_k_i);
                kx_q     <= 8'(g_t0_i % g_k_i);
                oy_q     <= g_oy0_i;
                ox_q     <= g_ox0_i;
                rows_q   <= rows_i;
                // Nothing to do still finishes, so a poll on done never hangs.
                state <= (rows_i != '0 && g_t1_i > g_t0_i && g_cw_i != '0 && g_k_i != '0)
                         ? S_TAP : S_DONE;
              end
              M_WB: begin
                rowbase_q <= {src_addr_i[31:4], 4'b0000};
                grpbase_q <= {src_addr_i[31:4], 4'b0000};
                src_q     <= {src_addr_i[31:4], 4'b0000};
                cols_q    <= wb_cols_i;
                col_q     <= '0;
                rstride_q <= wb_stride_i;
                gstride_q <= wb_grp_stride_i;
                grp_q     <= (wb_grp_i == '0) ? rows_i : wb_grp_i;
                gi_q      <= '0;
                rows_q    <= rows_i;
                state     <= (rows_i != '0 && wb_cols_i != '0) ? S_RD : S_DONE;
              end
              default: begin
                // The low four bits are dropped rather than honoured: a 128-bit
                // word straddling two source words would need a barrel shifter
                // on the read path for no benefit, since every producer of a
                // tile aligns it.
                src_q <= {src_addr_i[31:4], 4'b0000};
                rem_q <= len_i;
                if (len_i != '0) state <= S_REQ;
              end
            endcase
          end
        end

        // ---------------------------------------------------------- reads
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
          if (rem_q != 17'd1)      state <= S_REQ;
          else if (mode_q == M_GATHER) next_tap();
          else                     state <= S_DONE;
        end

        // --------------------------------------------------------- gather
        S_TAP: begin
          iy_q <= $signed({2'b0, oy_q}) * $signed({10'b0, stride_q}) + $signed({10'b0, ky_q})
                  - $signed({14'b0, pad_q});
          ix_q <= $signed({2'b0, ox_q}) * $signed({10'b0, stride_q}) + $signed({10'b0, kx_q})
                  - $signed({14'b0, pad_q});
          state <= S_ADDR;
        end

        S_ADDR: begin
          rem_q <= 17'(cw_q);
          if (tap_in) begin
            src_q <= tap_addr;
            state <= S_REQ;
          end else begin
            state <= S_ZERO;
          end
        end

        S_ZERO: begin
          wr_en_o     <= 1'b1;
          wr_region_o <= region_q;
          wr_addr_o   <= dst_q;
          wr_data_o   <= '0;
          dst_q       <= dst_q + DST_AW'(1);
          rem_q       <= rem_q - 17'd1;
          if (rem_q == 17'd1) next_tap();
        end

        // ----------------------------------------------------- write-back
        S_RD: begin
          // rd_addr_o is dst_q; the word arrives next cycle.
          state <= S_RDW;
        end

        S_RDW: begin
          sh_q   <= rd_data_i;
          lane_q <= '0;
          dst_q  <= dst_q + DST_AW'(1);
          state  <= S_WREQ;
        end

        S_WREQ: begin
          if (gnt) state <= S_WRSP;
        end

        S_WRSP: begin
          if (rsp_valid) begin
            if (tl_i.d_error) err_q <= 1'b1;
            src_q <= src_q + 32'd4;
            if (lane_q != LB'(LANES - 1)) begin
              lane_q <= lane_q + LB'(1);
              state  <= S_WREQ;
            end else if (col_q + 16'd1 != cols_q) begin
              col_q <= col_q + 16'd1;
              state <= S_RD;
            end else begin
              // End of a row: the next one starts a row stride on, or a
              // group stride on from the group's first row.
              col_q <= '0;
              if (rows_q == 16'd1) begin
                state <= S_DONE;
              end else if (gi_q + 16'd1 == grp_q) begin
                gi_q      <= '0;
                grpbase_q <= grpbase_q + gstride_q;
                rowbase_q <= grpbase_q + gstride_q;
                src_q     <= grpbase_q + gstride_q;
                state     <= S_RD;
              end else begin
                gi_q      <= gi_q + 16'd1;
                rowbase_q <= rowbase_q + rstride_q;
                src_q     <= rowbase_q + rstride_q;
                state     <= S_RD;
              end
              rows_q <= rows_q - 16'd1;
            end
          end
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
