#!/usr/bin/env python3
# ABOUTME: Non-interactive runner: loads an ELF over OpenOCD, streams hostio to stdout, exits with the program's return value.
# ABOUTME: flow's `sw_project run` needs a terminal and an xterm; this works under redirection and in scripts.

"""Run a program on the board and capture what it prints.

    python -u src/sw/project/tools/run_fpga.py [--elf PATH] [--timeout S]

Exits with the program's return value, or 124 on timeout. On timeout it
prints the DDR3 watchdog registers first, since a program that stops
answering is very often a CPU wedged on a bus request that will never be
answered -- in which case the debugger cannot say where it is, but these
registers can (docs/DEBUGGING.md section 3 in the superproject).

Deliberately not the interactive loop in flow/tools/openocd.py: that one
puts the terminal in raw mode and dies with "Inappropriate ioctl" under
redirection, and starts openocd inside an xterm. Print is flushed on every
chunk, because a buffered print on a run that then hangs is
indistinguishable from the hang (docs/LESSONS.md).
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

RVLAB = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(RVLAB))

from flow.tools.openocd import Hostio, OpenOcd            # noqa: E402
from flow.tools.riscv_debug_helper import reset_halt_rvlab_cpu   # noqa: E402

DDR_CTRL_BASE = 0x1F001000       # DDR_CTRL0_BASE_ADDR in src/sw/include/rvlab.h


def read_wdog(ocd: OpenOcd) -> str:
    """Decode ddr_ctrl.wdog_addr (+0x8) / wdog_stat (+0xc), per reggen/ddr_ctrl.h."""
    addr = ocd.readword(DDR_CTRL_BASE + 0x8)
    stat = ocd.readword(DDR_CTRL_BASE + 0xC)
    stall, src, outst, op = stat >> 16, (stat >> 8) & 0xFF, (stat >> 3) & 0x1F, stat & 7
    opname = {0: "PutFullData", 1: "PutPartialData", 4: "Get"}.get(op, str(op))
    tail = " (SATURATED -- never answered)" if stall == 0xFFFF else ""
    return (f"ddr watchdog: addr={addr:08x} {opname} source={src} "
            f"outstanding={outst} stalled={stall} cycles{tail}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--elf", type=Path,
                    default=RVLAB / "build/sw_project/build/sw.elf")
    ap.add_argument("--cfg", type=Path, default=RVLAB / "src/design/openocd/fpga.cfg")
    ap.add_argument("--timeout", type=float, default=600.0, help="seconds")
    ap.add_argument("--poll", type=float, default=0.05, help="hostio poll interval")
    ap.add_argument("--log", type=Path, default=Path("openocd.log"),
                    help="where OpenOCD's own output goes; load_image reports there")
    a = ap.parse_args()

    log = open(a.log, "w")
    ocd_proc = subprocess.Popen(["openocd", "-f", str(a.cfg)], stdout=log, stderr=log)
    time.sleep(1.5)
    try:
        with OpenOcd() as ocd:
            reset_halt_rvlab_cpu(ocd)
            ocd.cmd("lpriscv1.tap.0 arp_examine")
            reset_halt_rvlab_cpu(ocd)
            # The DMI reset-halt above halts the hart, but OpenOCD's own view of
            # the target can lag it, and load_image refuses to write memory to
            # a target it believes is running. Ask for the halt through OpenOCD
            # too, and do not proceed on anything but "halted".
            ocd.cmd("halt")
            state = ocd.cmd("lpriscv1.tap.0 curstate")
            print(f"target state: {state}", flush=True)
            if state != "halted":
                print("cannot load into a running target", flush=True)
                return 2
            ocd.cmd("tcl_trace off")
            ocd.hostio_clear()
            # load_image fails silently on a bad path and the board then runs
            # whatever the bitstream's BRAM init holds -- the course's
            # test_rvlab -- which looks like a run and is not yours. Print what
            # OpenOCD says about the load and the verify, every time.
            print(f"loading {a.elf}", flush=True)
            ocd.cmd(f"load_image {a.elf} 0 elf")
            ocd.cmd(f"verify_image {a.elf} 0 elf")
            # The tcl port returns nothing for these; the proof is in the log.
            log.flush()
            for line in open(a.log):
                if "downloaded" in line or "verified" in line or "rror" in line:
                    print("  " + line.rstrip(), flush=True)
            ocd.cmd("reg pc 0x80")
            ocd.cmd("riscv set_mem_access sysbus")
            ocd.cmd("resume")
            t0 = time.monotonic()
            while True:
                ocd.hostio_read()
                if ocd.readword(Hostio.FLAGS) & 1:
                    ocd.hostio_read()
                    rv = ocd.readword(Hostio.RETVAL)
                    print(f"\nexecution finished in {time.monotonic() - t0:.1f} s, "
                          f"return value {rv}", flush=True)
                    return rv
                if time.monotonic() - t0 > a.timeout:
                    print(f"\nTIMEOUT after {a.timeout:.0f} s", flush=True)
                    print(read_wdog(ocd), flush=True)
                    return 124
                time.sleep(a.poll)
    finally:
        ocd_proc.kill()
        ocd_proc.wait()


if __name__ == "__main__":
    sys.exit(main())
