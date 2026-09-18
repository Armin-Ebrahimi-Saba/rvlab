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
} tpu_desc_t;

#define BLOB_VERSION 2u

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

int tpu_run_blob(uint32_t blob_addr, int verify) {
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
     * the last token of a key tensor, say -- and the exporter assumes zeros. */
    {
        volatile uint32_t *arena = (volatile uint32_t *)(blob_addr + h->total);
        for (uint32_t w = 0; w < h->arena_bytes / 4u; w++) arena[w] = 0;
    }

    const volatile tpu_desc_t *descs = (const volatile tpu_desc_t *)(blob_addr + h->desc_off);
    unsigned long c_cfg = 0, c_run = 0, c_chk = 0, c_out = 0;
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

        if (d.a_dram && tpu_dma(d.a_dram, TPU_DMA_ACT, 0, d.a_words)) return 1;
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
        if (d.out_dram) {
            t0 = TPU_CYC();
            const volatile uint32_t *src = &TPU_OUT[(d.dst & 0xFFFFu) * 4u];
            uint32_t cols = d.out_cols / 4u;
            for (uint32_t r = 0; r < d.out_rows; r++) {
                volatile uint32_t *dst = (volatile uint32_t *)(d.out_dram + r * d.out_stride);
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
    printf("tinytpu: cycles: dma %lu (%lu bytes), config %lu, engine %lu, "
           "result copy %lu; verify %lu\n",
           tpu_dma_cycles, tpu_dma_bytes, c_cfg, c_run, c_out, c_chk);
    return failed ? 1 : 0;
}
