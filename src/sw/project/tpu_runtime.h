/* SPDX-License-Identifier: CC0-1.0
 * SPDX-FileCopyrightText: 2026 RVLab Student Project
 */
#ifndef TPU_RUNTIME_H
#define TPU_RUNTIME_H

#include <stdint.h>

/* Walks the descriptors of a TPU1 blob at `blob_addr` (sw/export_tpu.py).
 * With `verify` set, every descriptor that carries expected results is
 * compared and the first mismatches printed. Returns 0 when every op ran and
 * every check passed. */
/* Checks a TPU1 blob at blob_addr and zeroes its activation arena. Once per
 * blob; prints and returns 1 if the blob is not one this driver can run. */
int tpu_blob_open(uint32_t blob_addr);

/* Runs every descriptor once. With verify, checks each op that carries
 * expected bytes. Prints a PASS/FAIL verdict and the cycle breakdown. */
int tpu_blob_run(uint32_t blob_addr, int verify);

#endif
