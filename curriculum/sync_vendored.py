"""Sync vendor blocks (mimic_contract, dyck_worlds) into kernel files.

Usage: python3 curriculum/sync_vendored.py [--check]

--check mode exits non-zero if any kernel's vendored block differs from
its reference module (used in the push loop before every push).

Blocks:
  mimic_contract -> curriculum/kaggle_push/math_to_mimic.py (Phase-2)
  dyck_worlds    -> curriculum/kaggle_push/math_to_language.py (Phase-3)
"""

import sys
import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parent.parent

BLOCKS = [
    {
        "ref": ROOT / "curriculum" / "mimic_contract.py",
        "kernel": ROOT / "curriculum" / "kaggle_push" / "math_to_mimic.py",
        "start": "# >>> VENDOR (mimic_contract) — do not edit outside the reference module",
        "end": "# <<< VENDOR (mimic_contract) — do not edit outside the reference module",
    },
    {
        "ref": ROOT / "curriculum" / "dyck_worlds.py",
        "kernel": ROOT / "curriculum" / "kaggle_push" / "math_to_language.py",
        "start": "# >>> VENDOR (dyck_worlds) — do not edit outside the reference module",
        "end": "# <<< VENDOR (dyck_worlds) — do not edit outside the reference module",
    },
    {
        "ref": ROOT / "curriculum" / "mimic_contract.py",
        "kernel": ROOT / "curriculum" / "kaggle_push" / "fisher_v3_extract.py",
        "start": "# >>> VENDOR (mimic_contract) — do not edit outside the reference module",
        "end": "# <<< VENDOR (mimic_contract) — do not edit outside the reference module",
    },
    {
        "ref": ROOT / "curriculum" / "mimic_contract.py",
        "kernel": ROOT / "curriculum" / "kaggle_push" / "math_to_language_1b.py",
        "start": "# >>> VENDOR (mimic_contract) — do not edit outside the reference module",
        "end": "# <<< VENDOR (mimic_contract) — do not edit outside the reference module",
    },
]


def extract_ref(block: dict) -> str:
    lines = block["ref"].read_text().splitlines(keepends=True)
    out, inside = [], False
    for line in lines:
        if block["start"] in line:
            inside = True
            out.append(line)
            continue
        if block["end"] in line:
            out.append(line)
            inside = False
            continue
        if inside:
            out.append(line)
    return "".join(out).rstrip("\n")


def _pattern(start: str) -> re.Pattern:
    esc = re.escape(start)
    return re.compile(
        rf"{esc}.*?# <<< VENDOR \([^)]*\) — do not edit[^\n]*",
        re.DOTALL)


def sync(block: dict, check: bool) -> bool:
    payload = extract_ref(block)
    kernel = block["kernel"]
    src = kernel.read_text()
    pat = _pattern(block["start"])
    m = pat.search(src)
    if m is None:
        raise SystemExit(f"{kernel.name}: vendor block not found")
    if check:
        if m.group(0) != payload:
            print(f"MISMATCH {kernel.name}: re-run python3 "
                  f"curriculum/sync_vendored.py")
            return False
        print(f"{kernel.name}: vendor block in sync")
        return True
    new, n = pat.subn(payload, src, count=1)
    assert n == 1
    kernel.write_text(new)
    print(f"{kernel.name}: vendor block synced")
    return True


if __name__ == "__main__":
    check = "--check" in sys.argv
    ok = True
    for b in BLOCKS:
        try:
            ok = sync(b, check) and ok
        except SystemExit as e:
            print(e)
            ok = False
    sys.exit(0 if ok else 1)
