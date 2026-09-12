#!/usr/bin/env python3
"""Export a RealViformer PyTorch state dictionary to the native RVF format."""

from __future__ import annotations

import argparse
import hashlib
import struct
import warnings
from dataclasses import dataclass
from pathlib import Path

import torch


MAGIC = b"RVFWT001"
VERSION = 1
ENDIAN_TAG = 0x01020304
DTYPE_FLOAT32 = 1
HEADER = struct.Struct("<8sIIIIQ")
RECORD = struct.Struct("<IIIIQQ")
TABLE_ALIGNMENT = 8
DATA_ALIGNMENT = 64


def align(value: int, alignment: int) -> int:
    return (value + alignment - 1) // alignment * alignment


@dataclass
class TensorRecord:
    name: bytes
    shape: tuple[int, ...]
    data: bytes
    offset: int = 0

    @property
    def table_size(self) -> int:
        raw = RECORD.size + 8 * len(self.shape) + len(self.name)
        return align(raw, TABLE_ALIGNMENT)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--state-key", default="params")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="TypedStorage is deprecated")
        checkpoint = torch.load(
            args.checkpoint, map_location="cpu", weights_only=True
        )
    if args.state_key not in checkpoint:
        raise KeyError(f"checkpoint has no state dictionary {args.state_key!r}")

    records: list[TensorRecord] = []
    for name, tensor in sorted(checkpoint[args.state_key].items()):
        if tensor.dtype != torch.float32:
            raise TypeError(f"{name}: expected float32, found {tensor.dtype}")
        contiguous = tensor.detach().cpu().contiguous()
        records.append(
            TensorRecord(
                name=name.encode("utf-8"),
                shape=tuple(contiguous.shape),
                data=contiguous.numpy().tobytes(order="C"),
            )
        )

    table_end = HEADER.size + sum(record.table_size for record in records)
    data_offset = align(table_end, DATA_ALIGNMENT)
    data_size = 0
    for record in records:
        data_size = align(data_size, DATA_ALIGNMENT)
        record.offset = data_size
        data_size += len(record.data)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("wb") as output:
        output.write(
            HEADER.pack(
                MAGIC,
                VERSION,
                ENDIAN_TAG,
                len(records),
                0,
                data_offset,
            )
        )
        for record in records:
            record_start = output.tell()
            output.write(
                RECORD.pack(
                    len(record.name),
                    DTYPE_FLOAT32,
                    len(record.shape),
                    0,
                    record.offset,
                    len(record.data),
                )
            )
            for dimension in record.shape:
                output.write(struct.pack("<Q", dimension))
            output.write(record.name)
            output.write(b"\0" * (align(output.tell() - record_start, TABLE_ALIGNMENT) -
                                  (output.tell() - record_start)))

        output.write(b"\0" * (data_offset - output.tell()))
        data_start = output.tell()
        for record in records:
            desired = data_start + record.offset
            output.write(b"\0" * (desired - output.tell()))
            output.write(record.data)

    digest = hashlib.sha256(args.output.read_bytes()).hexdigest()
    print(
        f"wrote {len(records)} tensors, {data_size:,} data bytes, "
        f"{args.output.stat().st_size:,} total bytes"
    )
    print(f"sha256 {digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
