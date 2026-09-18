/* SPDX-License-Identifier: CC0-1.0
 * SPDX-FileCopyrightText: 2026 RVLab Student Project
 *
 * tiny-tpu from the CPU's side: the register window, the fast aperture, the
 * config-region layout and the opcodes. Shared by the self-test (main.c) and
 * the blob interpreter (tpu_runtime.c); the exporter in the superproject
 * (sw/export_tpu.py) lays tables out to exactly these numbers.
 */

#ifndef TPU_H
#define TPU_H

#include <stdint.h>
#include <rvlab.h>
#include <reggen/tinytpu.h>

/* Fast-aperture layout: the top two address bits pick the region. */
#define TPU_REGION(i)   ((volatile uint32_t *)(TINYTPU_CORE_BASE_ADDR + ((i) << 16)))
#define TPU_ACT         TPU_REGION(0)
#define TPU_WGT         TPU_REGION(1)
#define TPU_OUT         TPU_REGION(2)
#define TPU_CFG         TPU_REGION(3)

#define TPU_REG(off)    (*(volatile uint32_t *)(TINYTPU_REGS_BASE_ADDR + (off)))

#define TPU_STATUS_FIELD(f)                                       \
    ((TPU_REG(TINYTPU_STATUS_OFFSET) >> TINYTPU_STATUS_##f##_LSB) \
     & TINYTPU_STATUS_##f##_MASK)

/* Config-region layout, in 32-bit words from TPU_CFG. The GEMM's per-channel
 * constants own the lower three quarters; the top quarter holds the vector
 * unit's tables. */
#define TPU_TAB         12288u
#define TPU_LN_PAR      (TPU_TAB)             /* gamma, beta -- two words each */
#define TPU_EXP_LO      (TPU_TAB + 2048u)
#define TPU_EXP_HI      (TPU_TAB + 2048u + 256u)
#define TPU_UN_LUT      (TPU_TAB + 2048u + 512u)

#define TPU_OP_GEMM      0u
#define TPU_OP_SOFTMAX   1u
#define TPU_OP_LAYERNORM 2u
#define TPU_OP_QADD      3u
#define TPU_OP_UNARY     4u
#define TPU_OP_TRANSPOSE 5u

/* Regions as the DMA descriptor numbers them, matching the address decode. */
/* src_a / src_b carry a region as well as a word index, so an operand can be
 * named where it already is. */
#define TPU_SRC(region, word) \
    ((uint32_t)(word) | ((uint32_t)(region) << TINYTPU_SRC_A_REGION_LSB))

#define TPU_DMA_ACT 0u
#define TPU_DMA_WGT 1u
#define TPU_DMA_OUT 2u

/* Cycle counter, for the per-op accounting both users keep. */
#define TPU_CYC() ((unsigned long)read_csr("mcycle"))

/* One DMA into a buffer, from anywhere on the crossbar. Returns 0 on success;
 * on timeout or a bus error it prints and returns 1. Adds to the totals. */
int tpu_dma(uint32_t src, unsigned dst_region, unsigned dst_word, unsigned words);

/* Starts `opcode` and polls for done. Returns the spin count, 0 on timeout. */
unsigned tpu_run(unsigned opcode);

extern unsigned long tpu_dma_cycles, tpu_dma_bytes;

#endif /* TPU_H */
