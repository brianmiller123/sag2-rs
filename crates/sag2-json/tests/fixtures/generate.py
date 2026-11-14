#!/usr/bin/env python3.12
"""Regenerate the `sag2-json` codec fixtures from CPython 3.12.

The golden values are produced by the interpreter itself, never by the Rust
code under test:

* ``floats.jsonl``     -- ``repr(float)`` for hand-picked and random bit
  patterns, keeping the raw bits so the Rust test reconstructs the exact
  ``f64`` with ``f64::from_bits``.
* ``json_cases.jsonl`` -- ``compact_json`` output for the named cases built in
  ``tests/compact_json.rs``.

Run from the repository root:

    python3.12 crates/sag2-json/tests/fixtures/generate.py
"""

from __future__ import annotations

import json
import random
import struct
import sys
from pathlib import Path

FIXTURES = Path(__file__).resolve().parent
RANDOM_BITS = 100_000
SEED = 20261114

# Notation thresholds (decpt 16/17 and -3/-4), subnormals, extremes, signed zero.
HAND_PICKED_FLOATS = [
    0.0,
    -0.0,
    1.0,
    -1.0,
    0.1,
    0.5,
    2.5,
    1.5,
    100.0,
    2 / 3,
    1e-7,
    1e-6,
    1e-5,
    1e-4,
    1.234e-4,
    9.999999999999999e-5,
    1e15,
    1e16,
    9.999999999999998e15,
    1.0000000000000002e16,
    1234567890123456.0,
    12345678901234567.0,
    1e100,
    1e-100,
    1e300,
    1.5e300,
    5e-324,
    2.2250738585072014e-308,
    1.7976931348623157e308,
    2.2250738585072009e-308,
]

HAND_PICKED_BITS = [
    0x0000000000000000,  # 0.0
    0x8000000000000000,  # -0.0
    0x7FF0000000000000,  # inf
    0xFFF0000000000000,  # -inf
    0x7FF8000000000000,  # nan
    0xFFF8000000000000,  # nan with the sign bit set
    0x7FF0000000000001,  # nan payload
    0xFFF0000000000001,  # nan payload, sign bit set
    0x0000000000000001,  # smallest subnormal
    0x000FFFFFFFFFFFFF,  # largest subnormal
    0x0010000000000000,  # smallest normal
    0x7FEFFFFFFFFFFFFF,  # f64::MAX
    0xFFEFFFFFFFFFFFFF,  # f64::MIN
]


def float_cases():
    seen = set()
    for bits in HAND_PICKED_BITS:
        seen.add(bits)
        yield bits
    for value in HAND_PICKED_FLOATS:
        bits = struct.unpack("<Q", struct.pack("<d", value))[0]
        if bits in seen:
            continue
        seen.add(bits)
        yield bits
    generator = random.Random(SEED)
    for _ in range(RANDOM_BITS):
        bits = generator.getrandbits(64)
        if bits in seen:
            continue
        seen.add(bits)
        yield bits


def write_float_fixtures() -> tuple[Path, int]:
    path = FIXTURES / "floats.jsonl"
    count = 0
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for bits in float_cases():
            value = struct.unpack("<d", struct.pack("<Q", bits))[0]
            record = {"bits": bits, "repr": repr(value)}
            handle.write(json.dumps(record, separators=(",", ":")) + "\n")
            count += 1
    return path, count


def json_cases() -> dict[str, object]:
    cases: dict[str, object] = {}
    cases["flat_object"] = {"a": 1, "b": None, "c": True, "d": False}
    cases["empty_containers"] = {"object": {}, "array": []}
    cases["empty_string"] = {"s": ""}
    cases["string_escapes"] = {"s": 'a"b\\c\nd\re\tf\bg\fh\x00i\x01j\x1fk\x7fl/m'}
    cases["unicode_passthrough"] = {"s": "中文 😀 café ẞ \u2028\u2029"}
    cases["key_escapes"] = {'a"b': 1, "k\\2": 2}
    cases["nested"] = {"outer": {"inner": [1, 2, {"deep": "x"}]}, "list": [[], [{}], None]}
    cases["object_order"] = {"z": 1, "a": 2, "m": 3}
    updated = {"a": 1, "b": 2}
    updated["a"] = 3
    cases["key_order_after_update"] = updated
    cases["floats"] = [
        0.1,
        1.0,
        -0.0,
        1e16,
        1e-7,
        1e300,
        1.7976931348623157e308,
        5e-324,
        2.5e-10,
        123456789012345678.0,
    ]
    cases["special_floats"] = [float("nan"), float("inf"), float("-inf"), -0.0, 0.0]
    cases["ints"] = [0, -1, 9223372036854775807, -9223372036854775808]
    cases["bools_and_null"] = [True, False, None]
    return cases


def write_json_fixtures() -> tuple[Path, int]:
    path = FIXTURES / "json_cases.jsonl"
    count = 0
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for name, value in json_cases().items():
            compact = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
            record = {"case": name, "compact": compact}
            handle.write(
                json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
            )
            count += 1
    return path, count


def main() -> None:
    if sys.version_info < (3, 12):
        raise SystemExit(
            f"CPython >= 3.12 required (found {sys.version.split()[0]}): "
            "golden float bytes must come from the interpreter the reference "
            "implementation runs on"
        )
    for path, count in (write_float_fixtures(), write_json_fixtures()):
        print(f"wrote {path.name}: {count} lines, {path.stat().st_size} bytes")


if __name__ == "__main__":
    main()
