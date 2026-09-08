/* SPDX-License-Identifier: CC0-1.0
 * SPDX-FileCopyrightText: 2024 RVLab Contributors
 */

/* End-to-end check of the tiny-tpu peripheral.
 *
 * The CV32E40P fills the activation, weight and per-channel constant windows,
 * launches one GEMM, and compares the results against numbers produced by the
 * emulator in sw/kernels_int.py. The emulator stays the specification here just
 * as it does for the Verilator testbenches -- this test only moves the
 * comparison across the bus, so it covers the register map, the address decode
 * and the 32-bit/128-bit width change that the RTL testbenches cannot.
 *
 * Weights take the other path: instead of being stored word by word, they are
 * pulled in by tiny-tpu's own DMA over the student host port. The source here
 * is main BRAM rather than DDR3 only because the fast batch simulation builds
 * without the DDR3 model -- the descriptor is identical for a 0x8xxxxxxx
 * source, and the crossbar routes it by address without the DMA knowing.
 *
 * The vector ops follow the GEMM through the same register map: op.code picks
 * the kernel, ctrl.start launches it, and status.done ends it. Each one is
 * checked against the emulator too, and between them they cover both halves of
 * the config region's table decode.
 */

#include <stdio.h>
#include <stdint.h>
#include <rvlab.h>
#include <reggen/tinytpu.h>
#include <regaccess.h>

#include "gemm_vectors.h"
#include "block_vectors.h"

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

static void tpu_load(volatile uint32_t *dst, const uint32_t *src, unsigned n) {
    for (unsigned i = 0; i < n; i++) dst[i] = src[i];
}

/* Runs one op and waits for it. Returns the spin count, or 0 if the poll never
 * had to wait -- which the caller treats as a failure for the same reason the
 * GEMM poll does. */
static unsigned tpu_run(unsigned opcode) {
    unsigned spins = 0;
    TPU_REG(TINYTPU_OP_OFFSET) = opcode;
    TPU_REG(TINYTPU_CTRL_OFFSET) = 1u << TINYTPU_CTRL_CLR_DONE_LSB;
    TPU_REG(TINYTPU_CTRL_OFFSET) = 1u << TINYTPU_CTRL_START_LSB;
    while (!(TPU_STATUS_FIELD(DONE))) {
        if (++spins > 2000000u) return 0;
    }
    return spins;
}

/* Compares `n` int8 results, which is not the same as comparing whole words:
 * the last word of a run is usually partial and its padding bytes are whatever
 * the buffer already held. */
static unsigned tpu_check(const char *name, const uint32_t *want, unsigned n) {
    unsigned bad = 0;
    for (unsigned i = 0; i < n; i++) {
        uint8_t got = (uint8_t)(TPU_OUT[i / 4] >> (8 * (i % 4)));
        uint8_t exp = (uint8_t)(want[i / 4] >> (8 * (i % 4)));
        if (got != exp) {
            if (bad < 4)
                printf("tinytpu: %s[%u] got %02x expected %02x\n",
                       name, i, got, exp);
            bad++;
        }
    }
    if (bad) printf("tinytpu: FAIL %s, %u/%u elements\n", name, bad, n);
    return bad;
}

static int run_vector_ops(void) {
    unsigned bad = 0, spins;

    /* Tables first. gamma then beta per channel: the beta write commits the
     * pair, the same contract the GEMM's shift write has. */
    for (unsigned i = 0; i < VEC_LN_LEN; i++) {
        TPU_CFG[TPU_LN_PAR + 2 * i + 0] = vec_ln_gamma[i];
        TPU_CFG[TPU_LN_PAR + 2 * i + 1] = vec_ln_beta[i];
    }
    for (unsigned i = 0; i < 256; i++) {
        TPU_CFG[TPU_EXP_LO + i] = vec_exp_lo[i];
        TPU_CFG[TPU_EXP_HI + i] = vec_exp_hi[i];
        TPU_CFG[TPU_UN_LUT + i] = vec_un_lut[i];
    }

    /* Every op reads operand A from word 0 of the activation region, operand B
     * from word 0 of the weight region, and writes to word 0 of the results. */
    TPU_REG(TINYTPU_SRC_A_OFFSET) = TPU_SRC(TPU_DMA_ACT, 0);
    TPU_REG(TINYTPU_SRC_B_OFFSET) = TPU_SRC(TPU_DMA_WGT, 0);
    TPU_REG(TINYTPU_DST_OFFSET)   = 0;

    /* -- activation LUT -- */
    tpu_load(TPU_ACT, vec_un_x, (sizeof(vec_un_x) / 4));
    TPU_REG(TINYTPU_VEC_CFG_OFFSET) = VEC_N | (1u << 16);
    spins = tpu_run(TPU_OP_UNARY);
    if (!spins) { printf("tinytpu: FAIL unary did not complete\n"); return 1; }
    bad += tpu_check("unary", vec_un_y, VEC_N);

    /* -- residual add -- */
    tpu_load(TPU_ACT, vec_qa_a, (sizeof(vec_qa_a) / 4));
    tpu_load(TPU_WGT, vec_qa_b, (sizeof(vec_qa_b) / 4));
    TPU_REG(TINYTPU_VEC_MULT_OFFSET)   = VEC_QA_MULT_A;
    TPU_REG(TINYTPU_VEC_MULT_B_OFFSET) = VEC_QA_MULT_B;
    TPU_REG(TINYTPU_VEC_SHIFT_OFFSET)  =
          VEC_QA_SHIFT_A
        | (VEC_QA_SHIFT_B << TINYTPU_VEC_SHIFT_B_LSB);
    spins = tpu_run(TPU_OP_QADD);
    if (!spins) { printf("tinytpu: FAIL qadd did not complete\n"); return 1; }
    bad += tpu_check("qadd", vec_qa_y, VEC_N);

    /* -- softmax -- */
    tpu_load(TPU_ACT, vec_sm_x, (sizeof(vec_sm_x) / 4));
    TPU_REG(TINYTPU_VEC_CFG_OFFSET)   = VEC_SM_LEN | (VEC_SM_ROWS << 16);
    TPU_REG(TINYTPU_VEC_MULT_OFFSET)  = VEC_SM_MULT;
    TPU_REG(TINYTPU_VEC_SHIFT_OFFSET) = VEC_SM_SHIFT;
    spins = tpu_run(TPU_OP_SOFTMAX);
    if (!spins) { printf("tinytpu: FAIL softmax did not complete\n"); return 1; }
    bad += tpu_check("softmax", vec_sm_y, VEC_SM_ROWS * VEC_SM_LEN);

    /* -- layernorm -- */
    tpu_load(TPU_ACT, vec_ln_x, (sizeof(vec_ln_x) / 4));
    TPU_REG(TINYTPU_VEC_CFG_OFFSET)    = VEC_LN_LEN | (VEC_LN_ROWS << 16);
    TPU_REG(TINYTPU_VEC_MULT_OFFSET)   = VEC_LN_MULT;
    TPU_REG(TINYTPU_VEC_SHIFT_OFFSET)  = VEC_LN_SHIFT;
    TPU_REG(TINYTPU_VEC_EPS_LO_OFFSET) = VEC_LN_EPS_LO;
    TPU_REG(TINYTPU_VEC_EPS_HI_OFFSET) = VEC_LN_EPS_HI;
    spins = tpu_run(TPU_OP_LAYERNORM);
    if (!spins) { printf("tinytpu: FAIL layernorm did not complete\n"); return 1; }
    bad += tpu_check("layernorm", vec_ln_y, VEC_LN_ROWS * VEC_LN_LEN);

    if (bad) return 1;
    printf("tinytpu: PASS vector ops -- %u unary, %u qadd, %ux%u softmax,"
           " %ux%u layernorm, all bit-exact\n",
           VEC_N, VEC_N, VEC_SM_ROWS, VEC_SM_LEN, VEC_LN_ROWS, VEC_LN_LEN);
    return 0;
}

/* ------------------------------------------------------------------ block
 *
 * One transformer block, run op by op out of the descriptor table in
 * block_vectors.h. Everything below is bookkeeping: the interesting part is
 * that there is no arithmetic here at all, only staging.
 *
 * The result region is the block's register file -- every tensor lives at a
 * word offset in it, and the generator allocates those offsets the way a
 * compiler allocates registers. Both engines can read operands straight out of
 * it (src_a.region / src_b.region) and write results back into it at a base
 * (dst), so naming a slot in three registers is all an intermediate costs. The
 * only DMA left in the loop is the one that fetches weights, which is data the
 * chip genuinely does not have yet.
 *
 * The cycle breakdown printed at the end is the point of the exercise: it
 * separates the bus traffic from the arithmetic from the per-op setup, so the
 * next thing to attack is a measurement rather than a guess.
 */

static unsigned dma_spins_total;

/* Cycle accounting for the block. The spin counters answer "how many times did
 * the CPU look?", which conflates the bus round trip of the poll itself with
 * the work being waited on. mcycle answers the question step 2 actually asks:
 * where does the wall clock go. */
static unsigned long cyc_dma, cyc_run, cyc_cfg, cyc_chk;
static unsigned long dma_bytes;
#define CYC() ((unsigned long)read_csr("mcycle"))

/* One 128-bit-word-granular copy into a buffer region. `src` is a byte address
 * anywhere the main crossbar reaches, which includes tiny-tpu's own fast
 * aperture -- that is what makes region-to-region copies possible without the
 * CPU touching the data. */
static int tpu_dma(uint32_t src, unsigned dst_region, unsigned dst_word, unsigned words) {
    if (words == 0) return 0;
    unsigned long t0 = CYC();
    dma_bytes += words * 16u;
    TPU_REG(TINYTPU_DMA_SRC_OFFSET) = src;
    TPU_REG(TINYTPU_DMA_DST_OFFSET) =
        dst_word | (dst_region << TINYTPU_DMA_DST_REGION_LSB);
    TPU_REG(TINYTPU_DMA_LEN_OFFSET) = words;
    TPU_REG(TINYTPU_CTRL_OFFSET) = 1u << TINYTPU_CTRL_CLR_DMA_DONE_LSB;
    TPU_REG(TINYTPU_CTRL_OFFSET) = 1u << TINYTPU_CTRL_DMA_START_LSB;

    unsigned spins = 0;
    while (!(TPU_STATUS_FIELD(DMA_DONE))) {
        if (++spins > 2000000u) {
            printf("tinytpu: DMA timeout src 0x%08lx -> r%u w%u\n",
                   (unsigned long)src, dst_region, dst_word);
            return 1;
        }
    }
    if (TPU_STATUS_FIELD(DMA_ERR)) {
        printf("tinytpu: DMA error src 0x%08lx\n", (unsigned long)src);
        return 1;
    }
    dma_spins_total += spins;
    cyc_dma += CYC() - t0;
    return 0;
}

/* Compares one op's destination slot against the emulator, element by element:
 * the tail word of a tensor is padding and holds whatever was there before. */
static unsigned blk_check(const blk_op_t *o) {
    unsigned bad = 0;
    unsigned base = (unsigned)o->d_slot * 4u;
    for (unsigned i = 0; i < o->expect_n; i++) {
        uint8_t got = (uint8_t)(TPU_OUT[base + i / 4] >> (8 * (i % 4)));
        uint8_t exp = (uint8_t)(o->expect[i / 4] >> (8 * (i % 4)));
        if (got != exp) {
            if (bad < 3)
                printf("tinytpu: %s[%u] got %02x expected %02x\n",
                       o->name, i, got, exp);
            bad++;
        }
    }
    if (bad) {
        printf("tinytpu: FAIL %s, %u/%u elements\n", o->name, bad, o->expect_n);
        printf("  op=%u a_slot=%u a_words=%u\n", o->op, o->a_slot, o->a_words);
        printf("  d_slot=%u rows=%u len=%u\n", o->d_slot, o->rows, o->len);
        for (unsigned w = 0; w < 4; w++) {
            printf("  w%u act=%08lx\n", w, (unsigned long)TPU_ACT[w]);
            printf("  w%u src=%08lx\n", w,
                   (unsigned long)TPU_OUT[o->a_slot * 4u + w]);
            printf("  w%u dst=%08lx\n", w,
                   (unsigned long)TPU_OUT[o->d_slot * 4u + w]);
            printf("  w%u exp=%08lx\n", w, (unsigned long)o->expect[w]);
        }
    }
    return bad;
}

static int run_block(void) {
    unsigned run_spins = 0;

    dma_spins_total = 0;
    cyc_dma = cyc_run = cyc_cfg = cyc_chk = 0;
    dma_bytes = 0;

    /* Tables that outlive the whole block: the exponential pair is shared by
     * both heads' softmax because both use one logit scale, and the GELU table
     * is built at fc1's output scale so the activation needs no requantization
     * around it. LayerNorm's gamma/beta are the exception -- there is one RAM
     * and two norms, so it is reloaded per op below. */
    for (unsigned i = 0; i < 256; i++) {
        TPU_CFG[TPU_EXP_LO + i] = blk_exp_lo[i];
        TPU_CFG[TPU_EXP_HI + i] = blk_exp_hi[i];
        TPU_CFG[TPU_UN_LUT + i] = blk_un_lut[i];
    }

    /* The input tokens are the one tensor the CPU writes directly. */
    for (unsigned i = 0; i < BLK_X_WORDS * 4u; i++)
        TPU_OUT[BLK_X_SLOT * 4u + i] = blk_x[i];

    for (unsigned n = 0; n < BLK_N_OPS; n++) {
        const blk_op_t *o = &blk_ops[n];

        /* Operand A is always another op's output, so it is read in place:
         * naming the slot replaces a DMA of the whole tensor. Operand B is
         * either a weight tile, which genuinely has to be fetched, or another
         * intermediate, which does not. */
        TPU_REG(TINYTPU_SRC_A_OFFSET) = TPU_SRC(TPU_DMA_OUT, o->a_slot);

        if (o->wgt) {
            if (tpu_dma((uint32_t)(uintptr_t)o->wgt, TPU_DMA_WGT, 0, o->wgt_words))
                return 1;
            TPU_REG(TINYTPU_SRC_B_OFFSET) = TPU_SRC(TPU_DMA_WGT, 0);
        } else {
            TPU_REG(TINYTPU_SRC_B_OFFSET) = TPU_SRC(TPU_DMA_OUT, o->b_slot);
        }
        TPU_REG(TINYTPU_DST_OFFSET) = o->d_slot;

        unsigned long t_cfg = CYC();
        if (o->op == TPU_OP_GEMM) {
            for (unsigned c = 0; c < (unsigned)o->nt * 16u; c++) {
                TPU_CFG[4 * c + 0] = o->bias[c];
                TPU_CFG[4 * c + 1] = o->mult[c];
                TPU_CFG[4 * c + 2] = o->shift[c];
            }
            TPU_REG(TINYTPU_SHAPE_OFFSET) =
                  ((uint32_t)o->m)
                | ((uint32_t)o->kt << 12)
                | ((uint32_t)o->nt << 20);
        } else {
            if (o->ln_len) {
                for (unsigned i = 0; i < o->ln_len; i++) {
                    TPU_CFG[TPU_LN_PAR + 2 * i + 0] = o->ln_gamma[i];
                    TPU_CFG[TPU_LN_PAR + 2 * i + 1] = o->ln_beta[i];
                }
            }
            TPU_REG(TINYTPU_VEC_CFG_OFFSET) =
                (uint32_t)o->len | ((uint32_t)o->rows << 16);
            TPU_REG(TINYTPU_VEC_MULT_OFFSET)   = o->vmult;
            TPU_REG(TINYTPU_VEC_MULT_B_OFFSET) = o->vmult_b;
            TPU_REG(TINYTPU_VEC_SHIFT_OFFSET)  =
                  (uint32_t)o->vshift
                | ((uint32_t)o->vshift_b << TINYTPU_VEC_SHIFT_B_LSB);
            TPU_REG(TINYTPU_VEC_EPS_LO_OFFSET) = o->eps_lo;
            TPU_REG(TINYTPU_VEC_EPS_HI_OFFSET) = o->eps_hi;
        }

        cyc_cfg += CYC() - t_cfg;

        unsigned long t_run = CYC();
        unsigned spins = tpu_run(o->op);
        cyc_run += CYC() - t_run;
        if (!spins) {
            printf("tinytpu: FAIL %s did not complete\n", o->name);
            return 1;
        }
        run_spins += spins;

        unsigned long t_chk = CYC();
        unsigned bad = blk_check(o);
        cyc_chk += CYC() - t_chk;
        if (bad) return 1;
    }

    printf("tinytpu: PASS transformer block -- %u ops (%ux%u tokens, %u heads),"
           " every intermediate bit-exact (%u DMA spins, %u engine spins)\n",
           BLK_N_OPS, BLK_T, BLK_E, BLK_H, dma_spins_total, run_spins);

    /* The check is the testbench, not the block, so it is reported apart from
     * the three costs a real inference would still pay. */
    unsigned long work = cyc_dma + cyc_run + cyc_cfg;
    printf("tinytpu: cycles: dma %lu, engine %lu, config %lu (sum %lu)\n",
           cyc_dma, cyc_run, cyc_cfg, work);
    printf("tinytpu: dma moved %lu bytes -> %lu cycles/byte;"
           " check overhead %lu cycles\n",
           dma_bytes, dma_bytes ? cyc_dma / dma_bytes : 0, cyc_chk);
    return 0;
}

int main(void) {
    uint32_t id = TPU_REG(TINYTPU_ID_OFFSET);
    if (id != 0x54505530u) {
        printf("tinytpu: bad id 0x%08lx\n", (unsigned long)id);
        return 1;
    }
    printf("tinytpu: id ok\n");

    /* The standalone tests keep every operand at word 0 of its own region --
     * activations in, weights in, results out -- which is the reset placement.
     * Writing it anyway keeps the test independent of the reset values. */
    TPU_REG(TINYTPU_SRC_A_OFFSET) = TPU_SRC(TPU_DMA_ACT, 0);
    TPU_REG(TINYTPU_SRC_B_OFFSET) = TPU_SRC(TPU_DMA_WGT, 0);
    TPU_REG(TINYTPU_DST_OFFSET)   = 0;

    tpu_load(TPU_ACT, gemm_act, sizeof(gemm_act) / 4);

    /* Weights by DMA: three register writes replace sizeof(gemm_wgt)/4 stores. */
    TPU_REG(TINYTPU_DMA_SRC_OFFSET) = (uint32_t)(uintptr_t)gemm_wgt;
    TPU_REG(TINYTPU_DMA_DST_OFFSET) = TPU_DMA_WGT << TINYTPU_DMA_DST_REGION_LSB;
    TPU_REG(TINYTPU_DMA_LEN_OFFSET) = sizeof(gemm_wgt) / 16;
    TPU_REG(TINYTPU_CTRL_OFFSET) = 1u << TINYTPU_CTRL_CLR_DMA_DONE_LSB;
    TPU_REG(TINYTPU_CTRL_OFFSET) = 1u << TINYTPU_CTRL_DMA_START_LSB;

    unsigned dma_spins = 0;
    while (!(TPU_STATUS_FIELD(DMA_DONE))) {
        if (++dma_spins > 2000000u) {
            printf("tinytpu: DMA timed out, status 0x%08lx\n",
                   (unsigned long)TPU_REG(TINYTPU_STATUS_OFFSET));
            return 1;
        }
    }
    if (TPU_STATUS_FIELD(DMA_ERR)) {
        printf("tinytpu: FAIL DMA saw a bus error response\n");
        return 1;
    }
    if (dma_spins == 0) {
        printf("tinytpu: FAIL dma_done was already set; the poll did not wait\n");
        return 1;
    }

    /* Read one weight word back through the CPU aperture. The GEMM result would
     * catch a wholesale DMA failure, but not a buffer that the CPU and the DMA
     * disagree about the addressing of -- that would show up as a wrong answer
     * with no hint as to which side is wrong. */
    if (TPU_WGT[0] != gemm_wgt[0]) {
        printf("tinytpu: FAIL DMA wrote 0x%08lx at wgt[0], expected 0x%08lx\n",
               (unsigned long)TPU_WGT[0], (unsigned long)gemm_wgt[0]);
        return 1;
    }

    /* Four words per channel; the shift write commits the triple, so the order
     * below matters. The fourth word is padding that keeps the stride a power
     * of two, which turns the address arithmetic into a shift. */
    for (unsigned c = 0; c < GEMM_NT * 16; c++) {
        TPU_CFG[4 * c + 0] = gemm_bias[c];
        TPU_CFG[4 * c + 1] = gemm_mult[c];
        TPU_CFG[4 * c + 2] = gemm_shift[c];
    }

    TPU_REG(TINYTPU_OP_OFFSET) = 0;
    TPU_REG(TINYTPU_SHAPE_OFFSET) =
          ((uint32_t)GEMM_M)
        | ((uint32_t)GEMM_KT << 12)
        | ((uint32_t)GEMM_NT << 20);

    TPU_REG(TINYTPU_CTRL_OFFSET) = 1u << TINYTPU_CTRL_CLR_DONE_LSB;
    TPU_REG(TINYTPU_CTRL_OFFSET) = 1u << TINYTPU_CTRL_START_LSB;

    /* reggen's *_MASK is the field width, not a mask in place -- STATUS_DONE_MASK
     * is 0x1 with STATUS_DONE_LSB 0x1. Testing `status & DONE_MASK` therefore
     * reads *busy*, so the loop would fall through the instant the engine
     * started and the comparison below would race it. */
    unsigned spins = 0;
    while (!(TPU_STATUS_FIELD(DONE))) {
        if (++spins > 2000000u) {
            printf("tinytpu: timed out, status 0x%08lx\n",
                   (unsigned long)TPU_REG(TINYTPU_STATUS_OFFSET));
            return 1;
        }
    }

    unsigned words = GEMM_M * GEMM_NT * 4;
    unsigned bad = 0;
    for (unsigned i = 0; i < words; i++) {
        uint32_t got = TPU_OUT[i];
        if (got != gemm_expect[i]) {
            if (bad < 4)
                printf("tinytpu: word %u got 0x%08lx expected 0x%08lx\n",
                       i, (unsigned long)got, (unsigned long)gemm_expect[i]);
            bad++;
        }
    }

    if (bad) {
        printf("tinytpu: FAIL %u/%u words\n", bad, words);
        return 1;
    }
    /* A zero spin count would mean the poll fell through and the comparison
     * raced the engine, which is exactly how an earlier version of this test
     * passed for the wrong reason. */
    if (spins == 0) {
        printf("tinytpu: FAIL done was already set; the poll did not wait\n");
        return 1;
    }
    printf("tinytpu: PASS %ux%ux%u GEMM, %u words bit-exact vs qlinear()"
           " (%u DMA spins, %u poll spins)\n",
           GEMM_M, GEMM_K, GEMM_N, words, dma_spins, spins);

    /* The vector ops reuse all three buffers, so they only run once the GEMM
     * result has been read back and checked. */
    if (run_vector_ops()) return 1;

    return run_block();
}
