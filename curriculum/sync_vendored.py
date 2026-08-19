"""Sync the mimic_contract vendor block into the Phase-2 kernel file.

Usage: python3 curriculum/sync_vendored.py [--check]

--check mode exits non-zero if the kernel's vendored block differs from
the reference module (used in the push loop before every push).
"""

import sys
import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parent.parent
REF = ROOT / "curriculum" / "mimic_contract.py"
KERNEL = ROOT / "curriculum" / "kaggle_push" / "math_to_mimic.py"

START = "# >>> VENDOR (mimic_contract) — do not edit outside the reference module"
END = "# <<< VENDOR (mimic_contract) — do not edit outside the reference module"


def extract_ref() -> str:
    lines = REF.read_text().splitlines(keepends=True)
    out, inside = [], False
    for line in lines:
        if START in line:
            inside = True
            out.append(line)
            continue
        if END in line:
            out.append(line)
            inside = False
            continue
        if inside:
            out.append(line)
    return "".join(out)


def patch_kernel(block: str) -> bool:
    src = KERNEL.read_text()
    pattern = re.compile(
        r"# >>> VENDOR \(mimic_contract\).*?# <<< VENDOR \(mimic_contract\)[^\n]*",
        re.DOTALL)
    new, n = pattern.subn(block, src, count=1)
    if n != 1:
        raise SystemExit("kernel vendor block not found")
    KERNEL.write_text(new)
    return True


if __name__ == "__main__":
    block = extract_ref()
    if "--check" in sys.argv:
        src = KERNEL.read_text()
        pattern = re.compile(
            r"# >>> VENDOR \(mimic_contract\).*?# <<< VENDOR \(mimic_contract\)[^\n]*",
            re.DOTALL)
        cur = pattern.search(src)
        if cur is None or cur.group(0) != block:
            print("MISMATCH: re-run python3 curriculum/sync_vendored.py")
            sys.exit(1)
        print("vendor block in sync")
    else:
        patch_kernel(block)
        print(f"kernel vendor block synced from {REF.name}")