// SPDX-License-Identifier: CC0-1.0
// SPDX-FileCopyrightText: 2026 RVLab Student Project
//
// Stalled-transaction watchdog for a TL-UL device port.
//
// When cv32e40p issues a load whose response never arrives, it can never
// retire that instruction, so it can never enter debug mode either: the
// debug module reports the hart as running and every register read comes back
// empty. There is no pc to fetch, and no software instrumentation can help --
// the program is not executing.
//
// What does still work on a wedged system is JTAG system-bus access, which is
// how the accelerator's debug registers and the console ring stay readable.
// So this module records, in hardware, what the debugger cannot ask the core:
// the address of the oldest request that has not been answered, which master
// issued it, and how long it has been waiting.
//
// It is a pure observer. It drives nothing on the bus and cannot perturb the
// traffic it watches, which matters when the bug being chased is a lost
// handshake.
//
// Instantiated in rvlab_tlul_ddr.sv on the DDR3 TL-UL port; the two outputs
// are the ddr_ctrl registers wdog_addr and wdog_stat -- appended after ctrl
// in src/design/reggen/ddr_ctrl.hjson so no existing offset moves; take the
// addresses from the generated reggen/ddr_ctrl.h rather than a comment.
//
// wdog_stat packs {stall_cycles[15:0], src[7:0], outstanding[4:0],
// opcode[2:0]}. The source id says which master issued the request: on this
// SoC the low two bits index the crossbar's DDR3 hosts (0 CPU instruction,
// 1 CPU data, 2 debug module, 3 student host). This is what located a CPU
// hang on the sibling project this module comes from (docs/DEBUGGING.md
// section 3): one line read "PutFullData from the accelerator's write engine,
// 30 outstanding, stalled counter saturated" -- on a port that should never
// hold more than two.
//
// "Outstanding" means issued but not yet answered.  "Saturating" means the
// counter stops at its maximum instead of wrapping to zero, so a pinned
// maximum is unambiguous evidence of a hang.

module student_tl_watch #(
    // Requests may be outstanding this many at once before the counter
    // saturates. The DDR3 port is far shallower than this in practice.
    parameter int unsigned MAX_OUTSTANDING = 31
) (
    input  logic clk_i,
    input  logic rst_ni,

    // Snooped device port: what the bus presents to the DDR3 subsystem.
    input  tlul_pkg::tl_h2d_t tl_h2d_i,
    input  tlul_pkg::tl_d2h_t tl_d2h_i,

    output logic [31:0] wdog_addr_o,
    output logic [31:0] wdog_stat_o
);

  localparam int unsigned STALL_MAX = 16'hFFFF;

  wire a_beat = tl_h2d_i.a_valid & tl_d2h_i.a_ready;
  wire d_beat = tl_d2h_i.d_valid & tl_h2d_i.d_ready;

  logic [31:0] addr_q;
  logic [2:0]  opcode_q;
  logic [7:0]  src_q;
  logic [4:0]  outstanding_q;
  logic [15:0] stall_q;

  // The oldest unanswered request is approximated by the first request taken
  // while nothing was outstanding. That is exact for the single-outstanding
  // CPU, and for deeper traffic it still names a request that was in flight
  // when the stall began -- which is the question being asked.
  wire capture = a_beat & (outstanding_q == '0);

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      addr_q        <= '0;
      opcode_q      <= '0;
      src_q         <= '0;
      outstanding_q <= '0;
      stall_q       <= '0;
    end else begin
      if (capture) begin
        addr_q   <= tl_h2d_i.a_address;
        opcode_q <= tl_h2d_i.a_opcode;
        src_q    <= 8'(tl_h2d_i.a_source);
      end

      unique case ({a_beat, d_beat})
        2'b10:   if (outstanding_q != 5'(MAX_OUTSTANDING))
                   outstanding_q <= outstanding_q + 5'd1;
        2'b01:   if (outstanding_q != '0)
                   outstanding_q <= outstanding_q - 5'd1;
        default: ;
      endcase

      // Time since the port last had nothing outstanding. Saturating, so a
      // value pinned at the maximum is unambiguous evidence of a hang rather
      // than a wrapped counter that happens to look large.
      if (d_beat || (outstanding_q == '0))
        stall_q <= '0;
      else if (stall_q != 16'(STALL_MAX))
        stall_q <= stall_q + 16'd1;
    end
  end

  assign wdog_addr_o = addr_q;
  assign wdog_stat_o = {stall_q, src_q, outstanding_q, opcode_q};

endmodule
