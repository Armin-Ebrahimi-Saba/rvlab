#!/usr/bin/env python3
# ABOUTME: Puts the Depth Anything V2 weight blob into DDR3 on the board and proves it arrived.
# ABOUTME: Runs sw.elf, waits for BLOB_READY, load_image's the blob at 0x80000000, sets the go flag, compares checksums.

"""Load the model into DDR3 on the board.

    python -u src/sw/project/tools/load_model.py --blob path/to/dav2_weights.bin

The program on the board brings DDR3 up and prints BLOB_READY; this then
writes the blob over JTAG with load_image (raw binary at 0x80000000), sets the
BRAM flag the program spins on, and reads back the program's checksum of the
whole blob through the CPU's own path. The same checksum over the file, here,
is the comparison. Exit 0 only when they match.

The flag's address comes from the ELF's symbol table, not from a constant,
so a relink cannot move it out from under this script.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import time
from pathlib import Path

RVLAB = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(RVLAB))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from flow.tools.openocd import Hostio, OpenOcd                   # noqa: E402
from flow.tools.riscv_debug_helper import reset_halt_rvlab_cpu   # noqa: E402
from run_fpga import read_wdog                                   # noqa: E402

BLOB_ADDR = 0x80000000
BLOB_GO = 0x5B10B                     # BLOB_GO in main.c


def file_checksum(path: Path) -> tuple[int, int]:
    """The device's rotate-xor over whole words, from the header's total_bytes."""
    data = path.read_bytes()
    total = int.from_bytes(data[20:24], "little")
    words = memoryview(data)[: total & ~3].cast("I")
    s = 0
    for w in words:
        s = (((s << 1) | (s >> 31)) & 0xFFFFFFFF) ^ w
    return s, total


def symbol_address(elf: Path, name: str) -> int:
    out = subprocess.run(["riscv-none-elf-nm", str(elf)], capture_output=True,
                         text=True, check=True).stdout
    m = re.search(rf"^([0-9a-f]+) [bBdD] {re.escape(name)}$", out, re.M)
    if not m:
        raise SystemExit(f"{name} not in {elf}")
    return int(m.group(1), 16)


class Console:
    """Hostio drained into a buffer we can search, and echoed to stdout."""

    def __init__(self, ocd: OpenOcd):
        self.ocd, self.text = ocd, ""

    def pump(self) -> None:
        import io
        buf, real = io.StringIO(), sys.stdout
        sys.stdout = buf
        try:
            self.ocd.hostio_read()
        finally:
            sys.stdout = real
        chunk = buf.getvalue().replace("\r\n", "\n")
        if chunk:
            real.write(chunk)
            real.flush()
            self.text += chunk

    def wait_for(self, pattern: str, timeout: float) -> re.Match | None:
        t0 = time.monotonic()
        while time.monotonic() - t0 < timeout:
            self.pump()
            m = re.search(pattern, self.text)
            if m:
                return m
            if self.ocd.readword(Hostio.FLAGS) & 1:
                return None
            time.sleep(0.05)
        return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--blob", type=Path, required=True)
    ap.add_argument("--elf", type=Path, default=RVLAB / "build/sw_project/build/sw.elf")
    ap.add_argument("--cfg", type=Path, default=RVLAB / "src/design/openocd/fpga.cfg")
    ap.add_argument("--log", type=Path, default=Path("openocd.log"))
    a = ap.parse_args()

    go_addr = symbol_address(a.elf, "blob_go")
    want, total = file_checksum(a.blob)
    print(f"blob {a.blob}: {total} bytes, checksum {want:08x}; go flag at {go_addr:#010x}",
          flush=True)

    log = open(a.log, "w")
    proc = subprocess.Popen(["openocd", "-f", str(a.cfg)], stdout=log, stderr=log)
    time.sleep(1.5)
    try:
        with OpenOcd() as ocd:
            reset_halt_rvlab_cpu(ocd)
            ocd.cmd("lpriscv1.tap.0 arp_examine")
            reset_halt_rvlab_cpu(ocd)
            ocd.cmd("halt")
            if ocd.cmd("lpriscv1.tap.0 curstate") != "halted":
                print("cannot load into a running target", flush=True)
                return 2
            ocd.cmd("tcl_trace off")
            ocd.hostio_clear()
            ocd.cmd(f"load_image {a.elf} 0 elf")
            ocd.cmd(f"verify_image {a.elf} 0 elf")
            ocd.cmd("reg pc 0x80")
            ocd.cmd("riscv set_mem_access sysbus")
            ocd.cmd("resume")

            con = Console(ocd)
            if not con.wait_for(r"tinytpu: BLOB_READY", timeout=120):
                print("\nprogram never reported BLOB_READY", flush=True)
                print(read_wdog(ocd), flush=True)
                return 3

            # DDR3 is up and the program is spinning on the flag. Write the
            # blob. load_image is the fast path (~350 KiB/s over this cable);
            # verify_image would read it all back again, and the device-side
            # checksum is the check that matters, so it is not run here.
            t0 = time.monotonic()
            print(f"\nwriting {total} bytes to {BLOB_ADDR:#010x} ...", flush=True)
            ocd.cmd(f"load_image {a.blob} {BLOB_ADDR:#x} bin")
            log.flush()
            for line in open(a.log):
                if "downloaded" in line:
                    print("  " + line.rstrip(), flush=True)
            print(f"  {time.monotonic() - t0:.1f} s", flush=True)

            ocd.writeword(go_addr, BLOB_GO)

            m = con.wait_for(r"blob checksum \(device\) ([0-9a-f]{8}) over (\d+) bytes",
                             timeout=300)
            if not m:
                print("\nno checksum from the device", flush=True)
                print(read_wdog(ocd), flush=True)
                return 4
            got = int(m.group(1), 16)
            con.wait_for(r"\Z\A", timeout=2)          # drain the tail
            if got != want or int(m.group(2)) != total:
                print(f"\nMISMATCH: device {got:08x} over {m.group(2)}, "
                      f"file {want:08x} over {total}", flush=True)
                return 5
            print(f"\nmodel loaded: {total} bytes in DDR3, checksum {got:08x} matches the file",
                  flush=True)
            return 0
    finally:
        proc.kill()
        proc.wait()


if __name__ == "__main__":
    sys.exit(main())
