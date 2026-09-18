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
int tpu_run_blob(uint32_t blob_addr, int verify);

#endif
