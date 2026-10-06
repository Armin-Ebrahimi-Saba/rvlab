/* SPDX-License-Identifier: CC0-1.0
 * SPDX-FileCopyrightText: 2026 RVLab Student Project
 *
 * The blob interpreter: the program between sw/ and the hardware.
 *
 * Everything that could be decided offline has been, by sw/export_tpu.py in
 * the superproject: which bytes go in which buffer, at which word, in what
 * order, with what constants. What is left for this file is to walk a list
 * of descriptors and do, for each, the same six things run_block() in main.c
 * does for its 24 ops -- load operands, load config, set registers, start,
 * poll, check -- with the operands now coming from DDR3 instead of BRAM.
 *
 * That is deliberate. A tiling bug is then a Python bug, found by reading a
 * blob dump, and the C on this core stays small enough to be obviously right.
 * The cost is descriptor traffic, 96 bytes per op through the cache, which is
 * nothing next to the operands.
 *
 * The blob's format is documented at the top of sw/export_tpu.py; this file
 * is the other half of that contract and must not drift from it.
 */

#include <stdio.h>
#include <stdint.h>
#include "tpu.h"
#include "tpu_runtime.h"

#define OP_CONFIG 0x10u
#define OP_END    0x11u

typedef struct {
    uint32_t op, a_dram, a_words, b_dram, b_words, src_a, src_b, dst, shape;
    uint32_t cfg_dram, cfg_dst, cfg_n;
    uint32_t vmult, vmult_b, vshift, eps_lo, eps_hi;
    uint32_t out_dram, out_cols, out_rows, out_stride, check_dram, check_n, name_off;
    uint32_t g_src, g_hw, g_cw, g_k, g_p0, out_grp, out_grp_stride, pad;
} tpu_desc_t;

#define BLOB_VERSION 3u

/* Builds a GEMM tile's activation operand as im2col rows, straight into the
 * activation buffer: row i is output pixel g_p0 + i, and holds kernel taps
 * t0..t1-1 in row-major order, each the C bytes of one input pixel, or zeros
 * where the tap falls in the padding. C is a multiple of 16, so every tap is
 * whole buffer words and the copy is 32-bit throughout.
 *
 * This is the one place the driver computes addresses rather than reading
 * them: unrolled into descriptors, a 126x126 3x3 conv would be 143,000 DMAs.
 * The rows are exactly sw/export_head.py's im2col(), which the expected
 * values are computed from. */
static void gather_im2col(const tpu_desc_t *d, uint32_t rows) {
    uint32_t H = d->g_hw & 0xFFFFu, W = d->g_hw >> 16;
    uint32_t C = d->g_cw & 0xFFFFu, Wo = d->g_cw >> 16;
    uint32_t k = d->g_k & 0xFFu, stride = (d->g_k >> 8) & 0xFFu, pad = (d->g_k >> 16) & 0xFu;
    uint32_t t0 = (d->g_k >> 20) & 0x3Fu, t1 = (d->g_k >> 26) & 0x3Fu;
    uint32_t cw = C / 4u;                              /* u32 words per pixel */
    volatile uint32_t *dst = TPU_ACT;
    uint32_t oy = d->g_p0 / Wo, ox = d->g_p0 % Wo;
    for (uint32_t i = 0; i < rows; i++) {
        for (uint32_t t = t0; t < t1; t++) {
            int32_t iy = (int32_t)(oy * stride + t / k) - (int32_t)pad;
            int32_t ix = (int32_t)(ox * stride + t % k) - (int32_t)pad;
            if (iy < 0 || ix < 0 || iy >= (int32_t)H || ix >= (int32_t)W) {
                for (uint32_t w = 0; w < cw; w++) *dst++ = 0;
            } else {
                const volatile uint32_t *src = (const volatile uint32_t *)
                    (d->g_src + ((uint32_t)iy * W + (uint32_t)ix) * C);
                for (uint32_t w = 0; w < cw; w++) *dst++ = src[w];
            }
        }
        if (++ox == Wo) { ox = 0; oy++; }
    }
}

/* The same rows, built by the DMA's gather mode (tinytpu_wdma.sv): the
 * descriptor's geometry goes to the registers nearly verbatim -- g_k has the
 * register's packing -- and only the first pixel is split into (oy, ox). */
static int gather_dma(const tpu_desc_t *d, uint32_t rows) {
    uint32_t C = d->g_cw & 0xFFFFu, Wo = d->g_cw >> 16;
    TPU_REG(TINYTPU_DMA_MODE_OFFSET)  = TPU_DMA_MODE_GATHER;
    TPU_REG(TINYTPU_DMA_SRC_OFFSET)   = d->g_src;
    TPU_REG(TINYTPU_DMA_DST_OFFSET)   = TPU_SRC(TPU_DMA_ACT, 0);
    TPU_REG(TINYTPU_DMA_ROWS_OFFSET)  = rows;
    TPU_REG(TINYTPU_DMA_G_HW_OFFSET)  = d->g_hw;
    TPU_REG(TINYTPU_DMA_G_CW_OFFSET)  = (C / 16u) | (Wo << 16);
    TPU_REG(TINYTPU_DMA_G_K_OFFSET)   = d->g_k;
    TPU_REG(TINYTPU_DMA_G_O0_OFFSET)  = (d->g_p0 / Wo) | ((d->g_p0 % Wo) << 16);
    return tpu_dma_go("gather");
}

/* The result tile out of OUT and into its tensor, by the DMA's write-back
 * mode: rows of whole buffer words, a row stride, and a group stride. */
static int writeback_dma(const tpu_desc_t *d) {
    TPU_REG(TINYTPU_DMA_MODE_OFFSET)       = TPU_DMA_MODE_WRITEBACK;
    TPU_REG(TINYTPU_DMA_SRC_OFFSET)        = d->out_dram;
    TPU_REG(TINYTPU_DMA_DST_OFFSET)        = TPU_SRC(TPU_DMA_OUT, d->dst & 0xFFFFu);
    TPU_REG(TINYTPU_DMA_ROWS_OFFSET)       = d->out_rows;
    TPU_REG(TINYTPU_DMA_WB_COLS_OFFSET)    = d->out_cols / 16u;
    TPU_REG(TINYTPU_DMA_WB_STRIDE_OFFSET)  = d->out_stride;
    TPU_REG(TINYTPU_DMA_WB_GRP_OFFSET)     = d->out_grp;
    TPU_REG(TINYTPU_DMA_WB_GSTRIDE_OFFSET) = d->out_grp_stride;
    return tpu_dma_go("write-back");
}

/* Which paths move data. Both are kept so one build can compare them: the
 * CPU paths are the reference the DMA's were checked against on the board. */
int tpu_use_dma_paths = 1;

/* After this many failed ops the rest are not worth the hostio bytes: every
 * op downstream of a wrong result is wrong too. */
#define MAX_FAILED_OPS 4u

typedef struct {
    char     magic[4];
    uint32_t version, n_desc, desc_off, names_off, data_off, total, base, arena_bytes;
} tpu_header_t;

/* Compares `n` int8 results at OUT[dst..] against the bytes at `want`, and
 * reports the first few mismatches. The tail of a partial word is padding. */
static unsigned check(const tpu_desc_t *d, const char *name) {
    const volatile uint8_t *want = (const volatile uint8_t *)d->check_dram;
    unsigned base = (d->dst & 0xFFFFu) * 4u, bad = 0;
    for (uint32_t i = 0; i < d->check_n; i++) {
        uint8_t got = (uint8_t)(TPU_OUT[base + i / 4] >> (8 * (i % 4)));
        uint8_t exp = want[i];
        if (got != exp) {
            if (bad < 3)
                printf("tinytpu: %s[%lu] got %02x expected %02x\n",
                       name, (unsigned long)i, got, exp);
            bad++;
        }
    }
    if (bad)
        printf("tinytpu: FAIL %s, %u/%lu elements\n", name, bad, (unsigned long)d->check_n);
    return bad;
}

int tpu_blob_open(uint32_t blob_addr) {
    const volatile tpu_header_t *h = (const volatile tpu_header_t *)blob_addr;
    if (h->magic[0] != 'T' || h->magic[1] != 'P' || h->magic[2] != 'U' || h->magic[3] != '1') {
        printf("tinytpu: not a TPU1 blob at %08lx\n", (unsigned long)blob_addr);
        return 1;
    }
    if (h->base != blob_addr) {
        printf("tinytpu: blob was exported for %08lx, loaded at %08lx\n",
               (unsigned long)h->base, (unsigned long)blob_addr);
        return 1;
    }
    if (h->version != BLOB_VERSION) {
        printf("tinytpu: blob format v%lu, this driver reads v%u\n",
               (unsigned long)h->version, BLOB_VERSION);
        return 1;
    }
    printf("tinytpu: blob v%lu, %lu descriptors, %lu bytes\n",
           (unsigned long)h->version, (unsigned long)h->n_desc, (unsigned long)h->total);

    /* The arena above the blob is where ops leave activations for each other.
     * Some of it is padding that is read but never written -- the rows past
     * the last token of a key tensor, say -- and the exporter assumes zeros.
     * Nothing ever writes the padding, so once per blob is enough, and the
     * host may then put an input into the arena before each run. */
    volatile uint32_t *arena = (volatile uint32_t *)(blob_addr + h->total);
    for (uint32_t w = 0; w < h->arena_bytes / 4u; w++) arena[w] = 0;
    return 0;
}

int tpu_blob_run(uint32_t blob_addr, int verify) {
    const volatile tpu_header_t *h = (const volatile tpu_header_t *)blob_addr;
    const volatile tpu_desc_t *descs = (const volatile tpu_desc_t *)(blob_addr + h->desc_off);
    unsigned long c_cfg = 0, c_run = 0, c_chk = 0, c_out = 0, c_gather = 0;
    unsigned n_wb = 0;
    unsigned ops = 0, failed = 0;
    tpu_dma_cycles = tpu_dma_bytes = 0;

    for (uint32_t i = 0; i < h->n_desc; i++) {
        tpu_desc_t d;
        /* One copy out of DRAM per descriptor: the fields are read many times
         * below and the cache line may well be gone by then. */
        for (unsigned w = 0; w < sizeof d / 4; w++)
            ((uint32_t *)&d)[w] = ((const volatile uint32_t *)&descs[i])[w];
        const char *name = d.name_off ? (const char *)d.name_off : "?";

        if (d.op == OP_END) break;

        if (d.g_src) {
            unsigned long tg = TPU_CYC();
            if (tpu_use_dma_paths) {
                if (gather_dma(&d, d.shape & 0xFFFu)) return 1;
            } else {
                gather_im2col(&d, d.shape & 0xFFFu);
            }
            c_gather += TPU_CYC() - tg;
        } else if (d.a_dram && tpu_dma(d.a_dram, TPU_DMA_ACT, 0, d.a_words)) {
            return 1;
        }
        if (d.b_dram && tpu_dma(d.b_dram, TPU_DMA_WGT, 0, d.b_words)) return 1;

        unsigned long t0 = TPU_CYC();
        if (d.cfg_n) {
            const volatile uint32_t *src = (const volatile uint32_t *)d.cfg_dram;
            for (uint32_t w = 0; w < d.cfg_n; w++)
                TPU_CFG[d.cfg_dst + w] = src[w];
        }
        if (d.op == OP_CONFIG) { c_cfg += TPU_CYC() - t0; continue; }

        TPU_REG(TINYTPU_SRC_A_OFFSET) = d.src_a;
        TPU_REG(TINYTPU_SRC_B_OFFSET) = d.src_b;
        TPU_REG(TINYTPU_DST_OFFSET)   = d.dst;
        if (d.op == TPU_OP_GEMM) {
            TPU_REG(TINYTPU_SHAPE_OFFSET) = d.shape;
        } else {
            TPU_REG(TINYTPU_VEC_CFG_OFFSET)    = d.shape;
            TPU_REG(TINYTPU_VEC_MULT_OFFSET)   = d.vmult;
            TPU_REG(TINYTPU_VEC_MULT_B_OFFSET) = d.vmult_b;
            TPU_REG(TINYTPU_VEC_SHIFT_OFFSET)  = d.vshift;
            TPU_REG(TINYTPU_VEC_EPS_LO_OFFSET) = d.eps_lo;
            TPU_REG(TINYTPU_VEC_EPS_HI_OFFSET) = d.eps_hi;
        }
        c_cfg += TPU_CYC() - t0;

        t0 = TPU_CYC();
        unsigned spins = tpu_run(d.op);
        c_run += TPU_CYC() - t0;
        ops++;
        if (!spins) {
            printf("tinytpu: FAIL %s did not complete (op %lu, shape %08lx)\n",
                   name, (unsigned long)d.op, (unsigned long)d.shape);
            return 1;
        }

        /* Results leave the chip through the CPU for now: one 32-bit load from
         * the aperture and one store to DRAM per word. The DMA only fills
         * buffers. This is the known cost of the first picture, not a design.
         * Rows are packed in the result region and strided in DRAM, so a tile
         * lands inside the tensor it belongs to. */
        if (d.out_dram && tpu_use_dma_paths && (d.out_cols % 16u) == 0 &&
            (d.out_dram % 16u) == 0 && (d.out_stride % 16u) == 0 &&
            (d.out_grp_stride % 16u) == 0) {
            t0 = TPU_CYC();
            if (writeback_dma(&d)) return 1;
            c_out += TPU_CYC() - t0;
            n_wb++;
        } else if (d.out_dram) {
            t0 = TPU_CYC();
            const volatile uint32_t *src = &TPU_OUT[(d.dst & 0xFFFFu) * 4u];
            uint32_t cols = d.out_cols / 4u;
            uint32_t grp = d.out_grp ? d.out_grp : d.out_rows;
            for (uint32_t r = 0; r < d.out_rows; r++) {
                uint32_t at = (r / grp) * d.out_grp_stride + (r % grp) * d.out_stride;
                volatile uint32_t *dst = (volatile uint32_t *)(d.out_dram + at);
                for (uint32_t w = 0; w < cols; w++) dst[w] = *src++;
            }
            c_out += TPU_CYC() - t0;
        }

        if (verify && d.check_n) {
            t0 = TPU_CYC();
            if (check(&d, name) && ++failed >= MAX_FAILED_OPS) {
                printf("tinytpu: giving up after %u failed ops\n", failed);
                break;
            }
            c_chk += TPU_CYC() - t0;
        }
    }

    printf("tinytpu: %s -- %u ops, %u failed\n", failed ? "FAIL blob" : "PASS blob",
           ops, failed);
    printf("tinytpu: cycles: dma %lu (%lu bytes), gather %lu, config %lu, engine %lu, "
           "result copy %lu; verify %lu; %s paths, %u write-backs\n",
           tpu_dma_cycles, tpu_dma_bytes, c_gather, c_cfg, c_run, c_out, c_chk,
           tpu_use_dma_paths ? "dma" : "cpu", n_wb);
    return failed ? 1 : 0;
}
