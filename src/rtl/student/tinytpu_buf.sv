// SPDX-License-Identifier: SHL-2.1
// SPDX-FileCopyrightText: 2024 RVLab Contributors

// One tiny-tpu data buffer, seen at two different widths.
//
// The array wants a whole 16-byte tile row per cycle; TL-UL is 32 bits wide and
// tlul_adapter_sram only speaks that width. So the buffer is built as four
// 32-bit banks sharing a word address: the CPU addresses {word, lane} and hits
// one bank, while the engine reads or writes all four at once and sees a
// 128-bit word.
//
// Reads and writes are separately addressed, so an engine can read a buffer it
// is also writing. That is what lets the result region be an operand region: an
// intermediate stays where the op that produced it left it, instead of being
// DMA'd back into the activation buffer first.
//
// A BRAM has two ports and this buffer needs more than two -- the CPU's, the
// engine's write, and one engine read per operand the buffer can serve. The
// extra read ports are bought by replicating the array, one copy per reader,
// with every write mirrored into all of them. That has to be written out
// explicitly: left to itself, Vivado does not replicate a three-port array, it
// dissolves the whole thing into flip-flops and a 512-deep mux tree, which for
// three of these buffers is a quarter of a million registers and does not fit
// on the part at all. Each copy here has exactly one write access and one read
// access, which is the shape a true dual-port BRAM has.
//
// Nothing arbitrates between the CPU and the engine, because nothing needs to:
// software fills the buffers while the engine is idle and reads results after
// `done`. Touching a buffer mid-run is a driver bug, not a hardware case to
// handle. Read-during-write to the same word returns the old value, which is
// what an in-place vector op relies on -- its reader is a whole word ahead of
// its packer by the time the packer commits.

module tinytpu_buf #(
  parameter int WORDS     = 256,             // 128-bit words
  parameter int LANES     = 4,               // 32-bit lanes per word
  parameter int WORD_BITS = 32 * LANES,
  // Engine read ports. The result buffer needs two, because a GEMM whose
  // stationary operand is another op's output reads both A and B from it.
  parameter int RPORTS    = 1
) (
  input  logic clk_i,

  // CPU side: 32 bits, addressed as {word, lane}
  input  logic                          cpu_req,
  input  logic                          cpu_we,
  input  logic [$clog2(WORDS*LANES)-1:0] cpu_addr,
  input  logic [31:0]                   cpu_wdata,
  input  logic [31:0]                   cpu_wmask,
  output logic [31:0]                   cpu_rdata,

  // Engine side: a full word per access, read and write independently addressed
  input  logic                          eng_we,
  input  logic [$clog2(WORDS)-1:0]      eng_waddr,
  input  logic [WORD_BITS/8-1:0]        eng_wmask,   // one bit per byte
  input  logic [WORD_BITS-1:0]          eng_wdata,
  input  logic [$clog2(WORDS)-1:0]      eng_raddr [RPORTS],
  output logic [WORD_BITS-1:0]          eng_rdata [RPORTS]
);

  localparam int LB = $clog2(LANES);
  localparam int WA = $clog2(WORDS);
  // One copy per reader: the CPU is copy 0, engine read port r is copy r+1.
  localparam int COPIES = RPORTS + 1;

  wire [LB-1:0] cpu_lane = cpu_addr[LB-1:0];
  wire [WA-1:0] cpu_word = cpu_addr[LB+WA-1:LB];

  logic [31:0] cpu_rd_lane [LANES];
  logic [31:0] eng_rd_lane [RPORTS][LANES];

  for (genvar c = 0; c < COPIES; c++) begin : g_copy
    for (genvar l = 0; l < LANES; l++) begin : g_bank
      logic [31:0] bank [WORDS];

      // Reading a never-written location must not put X on the bus. TL-UL's
      // adapters assert on unknown response data, and an X there does not stay
      // local -- it propagates through the crossbar FIFOs and takes down the
      // whole simulation. Xilinx BRAM honours this as an INIT string, so it
      // costs nothing in hardware either.
      initial for (int i = 0; i < WORDS; i++) bank[i] = '0;

      // One write port, shared. The CPU and the engine never write at once:
      // software fills the buffers while the engine is idle. Every copy takes
      // every write, which is what keeps them identical.
      always_ff @(posedge clk_i) begin
        if (cpu_req && cpu_we && cpu_lane == LB'(l)) begin
          for (int b = 0; b < 4; b++)
            if (cpu_wmask[8*b]) bank[cpu_word][8*b +: 8] <= cpu_wdata[8*b +: 8];
        end else if (eng_we) begin
          // The engine's byte mask exists for the vector unit: its last output
          // word of a run is usually partial, and a full-word write there would
          // clobber whatever follows it in the buffer.
          for (int b = 0; b < 4; b++)
            if (eng_wmask[4*l + b])
              bank[eng_waddr][8*b +: 8] <= eng_wdata[32*l + 8*b +: 8];
        end
      end

      // One read port, registered -- copy 0 serves the CPU, the rest serve one
      // engine read port each.
      if (c == 0) begin : g_cpu_rd
        // Default to zero and load only on a request, matching rvlab_bram_main.
        always_ff @(posedge clk_i) begin
          cpu_rd_lane[l] <= '0;
          if (cpu_req) cpu_rd_lane[l] <= bank[cpu_word];
        end
      end else begin : g_eng_rd
        always_ff @(posedge clk_i) eng_rd_lane[c-1][l] <= bank[eng_raddr[c-1]];
      end
    end
  end

  for (genvar r = 0; r < RPORTS; r++) begin : g_eng_word
    for (genvar l = 0; l < LANES; l++) begin : g_lane
      assign eng_rdata[r][32*l +: 32] = eng_rd_lane[r][l];
    end
  end

  // The lane select must be delayed with the data, since the read is registered.
  logic [LB-1:0] cpu_lane_q;
  always_ff @(posedge clk_i) cpu_lane_q <= cpu_lane;
  assign cpu_rdata = cpu_rd_lane[cpu_lane_q];

endmodule
