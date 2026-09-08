// SPDX-License-Identifier: SHL-2.1
// SPDX-FileCopyrightText: 2024 RVLab Contributors

// Student module -- tiny-tpu integration point.
//
// The rlight, dma and tlul_mux exercise modules were removed: tiny-tpu needs all
// three student ports for itself, and with the DMA gone it owns the single
// tl_host port outright, so no host-side arbiter is required.
//
// Port plan (see report.md section 5):
//   tl_device_peri -> tinytpu_regs  control / status / descriptors (reggen)
//   tl_device_fast -> tinytpu_core  activation SRAM + 16x16 int8 array + VPU
//   tl_host        -> tinytpu_wdma  weight streaming from DDR3 @0x80000000
//
// All three ports are live: tiny-tpu owns the student host port outright and
// drives it from its weight DMA.

module student (
  input logic clk_i,
  input logic rst_ni,

  input  top_pkg::userio_board2fpga_t userio_i,
  output top_pkg::userio_fpga2board_t userio_o,

  output logic irq_o,

  input  tlul_pkg::tl_h2d_t tl_device_peri_i,
  output tlul_pkg::tl_d2h_t tl_device_peri_o,
  input  tlul_pkg::tl_h2d_t tl_device_fast_i,
  output tlul_pkg::tl_d2h_t tl_device_fast_o,

  input  tlul_pkg::tl_d2h_t tl_host_i,
  output tlul_pkg::tl_h2d_t tl_host_o
);

  // userio is unused for now; the LEDs belonged to the removed rlight exercise.
  logic unused_userio;
  assign unused_userio = ^{userio_i};
  assign userio_o = '{default: '0};

  assign irq_o = '0;

  tinytpu tinytpu_i (
    .clk_i,
    .rst_ni,
    .tl_device_peri_i,
    .tl_device_peri_o,
    .tl_device_fast_i,
    .tl_device_fast_o,
    .tl_host_i,
    .tl_host_o
  );

endmodule
