// SPDX-License-Identifier: SHL-2.1
// SPDX-FileCopyrightText: 2024 RVLab Contributors

// tiny-tpu peripheral: the SoC-facing wrapper around the int8 GEMM engine.
//
// Two device ports, used for two different things:
//
//   tl_device_peri (0x10000000, slow xbar)  control and status, via reggen
//   tl_device_fast (0x20000000, fast xbar)  activations, weights, results and
//                                           the per-channel requant constants,
//                                           mapped as plain memory
//
// Splitting them this way is what keeps setup cheap. A 1536-channel GEMM needs
// a bias, a multiplier and a shift per output channel; as MMIO register writes
// that would be ~4600 accesses over the slow bus before any work starts. As a
// memory window the CPU just stores them, and the weight DMA fills the same
// window straight from DDR3 without the CPU in the loop at all.
//
// The fast aperture is split into four equal regions by the top two address
// bits: activations, weights, results, config.
//
// A third port faces outward: tl_host is the student host port on the main
// crossbar, and tinytpu_wdma uses it to fill any of the three data buffers
// straight from DDR3 (or from anywhere else the crossbar reaches) without the
// CPU moving the bytes.

module tinytpu #(
  parameter int ACT_WORDS = 512,   // 128-bit words
  parameter int WGT_WORDS = 512,
  parameter int OUT_WORDS = 512,
  parameter int ROWS      = 16,
  parameter int COLS      = 16,
  parameter int MBLK      = 16,
  parameter int MAX_M     = 2048,
  parameter int MAX_N     = 2048
) (
  input  logic clk_i,
  input  logic rst_ni,

  input  tlul_pkg::tl_h2d_t tl_device_peri_i,
  output tlul_pkg::tl_d2h_t tl_device_peri_o,
  input  tlul_pkg::tl_h2d_t tl_device_fast_i,
  output tlul_pkg::tl_d2h_t tl_device_fast_o,

  input  tlul_pkg::tl_d2h_t tl_host_i,
  output tlul_pkg::tl_h2d_t tl_host_o
);

  import tinytpu_reg_pkg::*;

  localparam int WORD_BITS = 8 * COLS;         // 128 bits at COLS = 16
  localparam int SRAM_AW   = 16;               // 64 Ki 32-bit words = 256 KiB aperture
  localparam int REG_AW    = SRAM_AW - 2;      // per-region word address
  localparam int ADDR_BITS = 20;               // gemm_seq address bus

  // -------------------------------------------------------------- registers
  tinytpu_reg2hw_t reg2hw;
  tinytpu_hw2reg_t hw2reg;

  tinytpu_reg_top u_regs (
    .clk_i,
    .rst_ni,
    .tl_i(tl_device_peri_i),
    .tl_o(tl_device_peri_o),
    .reg2hw,
    .hw2reg,
    .devmode_i(1'b1)
  );

  logic gemm_busy, gemm_done, done_sticky;
  logic vpu_busy, vpu_done;
  wire  busy = gemm_busy | vpu_busy;
  wire  done = gemm_done | vpu_done;

  wire start_pulse    = reg2hw.ctrl.start.qe    & reg2hw.ctrl.start.q;
  wire clr_done_pulse = reg2hw.ctrl.clr_done.qe & reg2hw.ctrl.clr_done.q;
  wire dma_start_pulse    = reg2hw.ctrl.dma_start.qe    & reg2hw.ctrl.dma_start.q;
  wire clr_dma_done_pulse = reg2hw.ctrl.clr_dma_done.qe & reg2hw.ctrl.clr_dma_done.q;

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni)          done_sticky <= 1'b0;
    else if (clr_done_pulse) done_sticky <= 1'b0;
    else if (done)           done_sticky <= 1'b1;
  end

  logic dma_busy, dma_done, dma_err;
  logic dma_done_sticky, dma_err_sticky;

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      dma_done_sticky <= 1'b0;
      dma_err_sticky  <= 1'b0;
    end else if (clr_dma_done_pulse) begin
      dma_done_sticky <= 1'b0;
      dma_err_sticky  <= 1'b0;
    end else begin
      if (dma_done) dma_done_sticky <= 1'b1;
      if (dma_err)  dma_err_sticky  <= 1'b1;
    end
  end

  assign hw2reg.id.d              = 32'h5450_5530;   // 'TPU0'
  assign hw2reg.status.busy.d     = busy;
  assign hw2reg.status.done.d     = done_sticky;
  assign hw2reg.status.dma_busy.d = dma_busy;
  assign hw2reg.status.dma_done.d = dma_done_sticky;
  assign hw2reg.status.dma_err.d  = dma_err_sticky;

  // ------------------------------------------------------------- data window
  logic                req, gnt, we;
  logic [SRAM_AW-1:0]  addr;
  logic [31:0]         wdata, wmask, rdata;
  logic                rvalid;

  assign gnt = 1'b1;   // the buffers accept an access every cycle

  tlul_adapter_sram #(
    .SramAw(SRAM_AW), .SramDw(32), .Outstanding(1), .ByteAccess(1)
  ) u_sram_adapter (
    .clk_i,
    .rst_ni,
    .tl_i(tl_device_fast_i),
    .tl_o(tl_device_fast_o),
    .req_o(req), .gnt_i(gnt), .we_o(we), .addr_o(addr),
    .wdata_o(wdata), .wmask_o(wmask),
    .rdata_i(rdata), .rvalid_i(rvalid), .rerror_i(2'b00)
  );

  typedef enum logic [1:0] { R_ACT, R_WGT, R_OUT, R_CFG } region_e;
  wire region_e       region  = region_e'(addr[SRAM_AW-1 -: 2]);
  wire [REG_AW-1:0]   roffset = addr[REG_AW-1:0];

  // The read is registered inside the buffers, so the region select has to be
  // delayed with it or the mux would pick using the *next* access's region.
  region_e region_q;
  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      rvalid   <= 1'b0;
      region_q <= R_ACT;
    end else begin
      rvalid   <= req & ~we;
      region_q <= region;
    end
  end

  logic [31:0] act_rdata, wgt_rdata, out_rdata;
  always_comb begin
    unique case (region_q)
      R_ACT:   rdata = act_rdata;
      R_WGT:   rdata = wgt_rdata;
      R_OUT:   rdata = out_rdata;
      default: rdata = '0;      // the config window is write-only
    endcase
  end

  // ---------------------------------------------------------------- buffers
  logic [ADDR_BITS-1:0] act_addr, wgt_addr, out_addr;
  logic [WORD_BITS-1:0] act_word, wgt_word, out_word;
  logic [WORD_BITS-1:0] act_word_r, wgt_word_r;
  logic [WORD_BITS-1:0] out_word_r [2];
  logic                 out_we;

  localparam int ACT_AW = $clog2(ACT_WORDS * 4);
  localparam int WGT_AW = $clog2(WGT_WORDS * 4);
  localparam int OUT_AW = $clog2(OUT_WORDS * 4);

  // The DMA shares the engine-side write port with the array. No arbiter: the
  // DMA runs while the array is idle, so the mux only has to pick, never
  // resolve. status.busy and status.dma_busy let the driver enforce that.
  localparam int DMA_AW = 16;

  logic                 dma_wr_en;
  logic [1:0]           dma_wr_region;
  logic [DMA_AW-1:0]    dma_wr_addr;
  logic [WORD_BITS-1:0] dma_wr_data;

  wire dma_wr_act = dma_wr_en && (region_e'(dma_wr_region) == R_ACT);
  wire dma_wr_wgt = dma_wr_en && (region_e'(dma_wr_region) == R_WGT);
  wire dma_wr_out = dma_wr_en && (region_e'(dma_wr_region) == R_OUT);

  // Both engines present the same three ports -- operand A, operand B, a
  // destination -- so which engine drives them is a two-way mux on `busy`.
  // Only one is ever busy: `start` dispatches on op.code.
  logic [ADDR_BITS-1:0] vpu_a_addr, vpu_b_addr, vpu_o_addr;
  logic [WORD_BITS-1:0] vpu_o_data;
  logic [WORD_BITS/8-1:0] vpu_o_wmask;
  logic                 vpu_o_we;

  wire [ADDR_BITS-1:0] eng_a_addr = vpu_busy ? vpu_a_addr : act_addr;
  wire [ADDR_BITS-1:0] eng_b_addr = vpu_busy ? vpu_b_addr : wgt_addr;

  // Operand A can come from the activation region or the result region, and
  // operand B from the weight region or the result region. That is what
  // src_a.region / src_b.region buy: an intermediate stays where the op that
  // produced it left it and is read in place, instead of being copied back into
  // an operand buffer by the DMA first. Both candidate buffers get the same
  // address and the data is muxed on the way back, which costs a mux and no
  // cycles -- every buffer read is registered with the same latency.
  wire a_from_out = (region_e'(reg2hw.src_a.region.q) == R_OUT);
  wire b_from_out = (region_e'(reg2hw.src_b.region.q) == R_OUT);

  assign act_word = a_from_out ? out_word_r[0] : act_word_r;
  assign wgt_word = b_from_out ? out_word_r[1] : wgt_word_r;

  logic [$clog2(ACT_WORDS)-1:0] act_raddr [1];
  logic [$clog2(WGT_WORDS)-1:0] wgt_raddr [1];
  logic [$clog2(OUT_WORDS)-1:0] out_raddr [2];
  logic [WORD_BITS-1:0]         act_rword [1];
  logic [WORD_BITS-1:0]         wgt_rword [1];

  assign act_raddr[0] = $clog2(ACT_WORDS)'(eng_a_addr);
  assign wgt_raddr[0] = $clog2(WGT_WORDS)'(eng_b_addr);
  assign out_raddr[0] = $clog2(OUT_WORDS)'(eng_a_addr);
  assign out_raddr[1] = $clog2(OUT_WORDS)'(eng_b_addr);
  assign act_word_r   = act_rword[0];
  assign wgt_word_r   = wgt_rword[0];

  tinytpu_buf #(.WORDS(ACT_WORDS)) u_act (
    .clk_i,
    .cpu_req(req && region == R_ACT), .cpu_we(we),
    .cpu_addr(ACT_AW'(roffset)), .cpu_wdata(wdata), .cpu_wmask(wmask),
    .cpu_rdata(act_rdata),
    .eng_we(dma_wr_act), .eng_wmask('1),
    .eng_waddr($clog2(ACT_WORDS)'(dma_wr_addr)),
    .eng_wdata(dma_wr_data),
    .eng_raddr(act_raddr), .eng_rdata(act_rword)
  );

  tinytpu_buf #(.WORDS(WGT_WORDS)) u_wgt (
    .clk_i,
    .cpu_req(req && region == R_WGT), .cpu_we(we),
    .cpu_addr(WGT_AW'(roffset)), .cpu_wdata(wdata), .cpu_wmask(wmask),
    .cpu_rdata(wgt_rdata),
    .eng_we(dma_wr_wgt), .eng_wmask('1),
    .eng_waddr($clog2(WGT_WORDS)'(dma_wr_addr)),
    .eng_wdata(dma_wr_data),
    .eng_raddr(wgt_raddr), .eng_rdata(wgt_rword)
  );

  tinytpu_buf #(.WORDS(OUT_WORDS), .RPORTS(2)) u_out (
    .clk_i,
    .cpu_req(req && region == R_OUT), .cpu_we(we),
    .cpu_addr(OUT_AW'(roffset)), .cpu_wdata(wdata), .cpu_wmask(wmask),
    .cpu_rdata(out_rdata),
    .eng_we(out_we | dma_wr_out | vpu_o_we),
    .eng_wmask(vpu_o_we ? vpu_o_wmask : '1),
    .eng_waddr(dma_wr_out ? $clog2(OUT_WORDS)'(dma_wr_addr)
             : vpu_o_we   ? $clog2(OUT_WORDS)'(vpu_o_addr)
                          : $clog2(OUT_WORDS)'(out_addr)),
    .eng_wdata(dma_wr_out ? dma_wr_data : vpu_o_we ? vpu_o_data : out_word),
    .eng_raddr(out_raddr), .eng_rdata(out_word_r)
  );

  // -------------------------------------------------------------- weight DMA
  tinytpu_wdma #(.WORD_BITS(WORD_BITS), .DST_AW(DMA_AW)) u_wdma (
    .clk_i,
    .rst_ni,
    .start_i     (dma_start_pulse),
    .src_addr_i  (reg2hw.dma_src.q),
    .dst_word_i  (DMA_AW'(reg2hw.dma_dst.word.q)),
    .dst_region_i(reg2hw.dma_dst.region.q),
    .len_i       (17'(reg2hw.dma_len.q)),
    .busy_o      (dma_busy),
    .done_o      (dma_done),
    .err_o       (dma_err),
    .wr_en_o     (dma_wr_en),
    .wr_region_o (dma_wr_region),
    .wr_addr_o   (dma_wr_addr),
    .wr_data_o   (dma_wr_data),
    .tl_o        (tl_host_o),
    .tl_i        (tl_host_i)
  );

  // --------------------------------------------------- per-channel constants
  // Three words per output channel: bias, multiplier, shift. The shift write is
  // the commit, so software must store them in that order -- which costs one
  // register pair here instead of a third write port on gemm_seq.
  logic                      cfg_we;
  logic [$clog2(MAX_N)-1:0]  cfg_addr;
  logic signed [31:0]        cfg_bias, cfg_mult;
  logic [5:0]                cfg_shift;

  wire cfg_write = req && we && (region == R_CFG);

  // The config region is split again. The top quarter holds the vector unit's
  // tables, which are bulk data like everything else on this port -- a 1024-entry
  // gamma/beta pair and three 256-entry lookup tables would be 1792 MMIO register
  // writes otherwise. The GEMM's per-channel constants keep the lower three
  // quarters, which is 3072 channels against a MAX_N of 2048.
  wire cfg_tables = cfg_write && (roffset[REG_AW-1:REG_AW-2] == 2'b11);
  wire cfg_chan   = cfg_write && (roffset[REG_AW-1:REG_AW-2] != 2'b11);

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      cfg_we <= 1'b0;
    end else begin
      cfg_we <= 1'b0;
      if (cfg_chan) begin
        unique case (roffset[1:0])
          2'd0: cfg_bias <= wdata;
          2'd1: cfg_mult <= wdata;
          2'd2: begin
            cfg_shift <= wdata[5:0];
            cfg_addr  <= $clog2(MAX_N)'(roffset[REG_AW-1:2]);
            cfg_we    <= 1'b1;
          end
          default: ;   // the fourth word pads each channel to a power of two
        endcase
      end
    end
  end

  // ------------------------------------------------- vector unit table loads
  //   roffset[11] == 0  layernorm gamma/beta, two words per channel
  //   roffset[11] == 1, roffset[10:8] == 0  exp LUT, low half
  //                                    1  exp LUT, high half
  //                                    2  activation LUT
  localparam int LN_LEN = 1024;
  localparam int SM_LEN = 1370;

  logic                       sm_lut_wr_en, sm_lut_wr_sel;
  logic [7:0]                 sm_lut_wr_addr;
  logic [15:0]                sm_lut_wr_data;
  logic                       un_lut_wr_en;
  logic [7:0]                 un_lut_wr_addr;
  logic signed [7:0]          un_lut_wr_data;
  logic                       ln_par_wr_en;
  logic [$clog2(LN_LEN)-1:0]  ln_par_wr_addr;
  logic signed [7:0]          ln_par_wr_gamma;
  logic signed [31:0]         ln_par_wr_beta;

  wire tab_is_par = cfg_tables && !roffset[11];
  wire tab_is_lut = cfg_tables &&  roffset[11];

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      sm_lut_wr_en <= 1'b0;
      un_lut_wr_en <= 1'b0;
      ln_par_wr_en <= 1'b0;
    end else begin
      sm_lut_wr_en <= 1'b0;
      un_lut_wr_en <= 1'b0;
      ln_par_wr_en <= 1'b0;

      // gamma first, beta second: the beta write commits the pair, the same
      // deal the GEMM's shift write makes.
      if (tab_is_par) begin
        if (!roffset[0]) begin
          ln_par_wr_gamma <= wdata[7:0];
        end else begin
          ln_par_wr_beta <= wdata;
          ln_par_wr_addr <= $clog2(LN_LEN)'(roffset[10:1]);
          ln_par_wr_en   <= 1'b1;
        end
      end

      if (tab_is_lut) begin
        unique case (roffset[10:8])
          3'd0, 3'd1: begin
            sm_lut_wr_sel  <= roffset[8];
            sm_lut_wr_addr <= roffset[7:0];
            sm_lut_wr_data <= wdata[15:0];
            sm_lut_wr_en   <= 1'b1;
          end
          3'd2: begin
            un_lut_wr_addr <= roffset[7:0];
            un_lut_wr_data <= wdata[7:0];
            un_lut_wr_en   <= 1'b1;
          end
          default: ;
        endcase
      end
    end
  end

  // ----------------------------------------------------------------- engine
  // ctrl.start dispatches on op.code: 0 is the GEMM, everything else is a
  // vector op. One engine runs at a time, which is what lets the buffer ports
  // be muxed rather than arbitrated.
  wire is_vec       = (reg2hw.op.q != 4'd0);
  wire gemm_start   = start_pulse && !is_vec;
  wire vpu_start    = start_pulse &&  is_vec;

  gemm_seq #(
    .ROWS(ROWS), .COLS(COLS), .MBLK(MBLK),
    .ADDR_BITS(ADDR_BITS), .MAX_M(MAX_M), .MAX_N(MAX_N)
  ) u_gemm (
    .clk(clk_i),
    .rst(~rst_ni),
    .cfg_wr_en(cfg_we), .cfg_wr_addr(cfg_addr),
    .cfg_wr_bias(cfg_bias), .cfg_wr_mult(cfg_mult), .cfg_wr_shift(cfg_shift),
    .start(gemm_start),
    .cfg_m($clog2(MAX_M+1)'(reg2hw.shape.m.q)),
    .cfg_k_tiles($clog2(MAX_N/COLS+1)'(reg2hw.shape.k_tiles.q)),
    .cfg_n_tiles($clog2(MAX_N/COLS+1)'(reg2hw.shape.n_tiles.q)),
    .cfg_a_word(reg2hw.src_a.word.q),
    .cfg_b_word(reg2hw.src_b.word.q),
    .cfg_d_word(reg2hw.dst.q),
    .busy(gemm_busy), .done(gemm_done),
    .act_addr(act_addr), .act_data(act_word),
    .wgt_addr(wgt_addr), .wgt_data(wgt_word),
    .out_we(out_we), .out_addr(out_addr), .out_data(out_word)
  );

  // ----------------------------------------------------------- vector unit
  vpu_seq #(
    .LANES(COLS), .SM_LEN(SM_LEN), .LN_LEN(LN_LEN), .ADDR_BITS(ADDR_BITS)
  ) u_vpu (
    .clk(clk_i),
    .rst(~rst_ni),
    .sm_lut_wr_en, .sm_lut_wr_sel, .sm_lut_wr_addr, .sm_lut_wr_data,
    .un_lut_wr_en, .un_lut_wr_addr, .un_lut_wr_data,
    .ln_par_wr_en, .ln_par_wr_addr, .ln_par_wr_gamma, .ln_par_wr_beta,
    .start(vpu_start),
    .op(reg2hw.op.q),
    .cfg_len(reg2hw.vec_cfg.len.q),
    .cfg_rows(reg2hw.vec_cfg.rows.q),
    .cfg_mult(reg2hw.vec_mult.q),
    .cfg_shift(reg2hw.vec_shift.a.q),
    .cfg_mult_b(reg2hw.vec_mult_b.q),
    .cfg_shift_b(reg2hw.vec_shift.b.q),
    .cfg_eps({reg2hw.vec_eps_hi.q, reg2hw.vec_eps_lo.q}),
    .cfg_a_word(reg2hw.src_a.word.q),
    .cfg_b_word(reg2hw.src_b.word.q),
    .cfg_d_word(reg2hw.dst.q),
    .busy(vpu_busy), .done(vpu_done),
    .a_addr(vpu_a_addr), .a_data(act_word),
    .b_addr(vpu_b_addr), .b_data(wgt_word),
    .o_we(vpu_o_we), .o_addr(vpu_o_addr),
    .o_wmask(vpu_o_wmask), .o_data(vpu_o_data)
  );

endmodule
