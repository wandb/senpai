# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai

"""Small VTK XML reader for the subset of AirfRANS files we actually use.

The student image does not ship `pyvista` or `meshio`, so AirfRANS needs a
repo-local parser. This module intentionally supports only the XML VTK surface
and volume files used by the dataset:

- `.vtu` and `.vtp`
- inline `ascii` arrays
- inline base64 `binary` arrays
- `vtkZLibDataCompressor` compressed binary arrays

That is enough to read AirfRANS points plus named point/cell arrays without
pulling in heavyweight third-party geometry stacks.
"""

from __future__ import annotations

import base64
import binascii
import zlib
from dataclasses import dataclass
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

VTK_TO_NUMPY = {
    "Float32": np.dtype("<f4"),
    "Float64": np.dtype("<f8"),
    "Int8": np.dtype("<i1"),
    "UInt8": np.dtype("<u1"),
    "Int16": np.dtype("<i2"),
    "UInt16": np.dtype("<u2"),
    "Int32": np.dtype("<i4"),
    "UInt32": np.dtype("<u4"),
    "Int64": np.dtype("<i8"),
    "UInt64": np.dtype("<u8"),
}

HEADER_TO_NUMPY = {
    "UInt32": np.dtype("<u4"),
    "UInt64": np.dtype("<u8"),
}


@dataclass(frozen=True)
class VTKPiece:
    points: np.ndarray
    point_data: dict[str, np.ndarray]
    cell_data: dict[str, np.ndarray]


def _local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _find_first_child(node: ET.Element, name: str) -> ET.Element | None:
    for child in node:
        if _local_name(child.tag) == name:
            return child
    return None


def _iter_children(node: ET.Element, name: str) -> list[ET.Element]:
    return [child for child in node if _local_name(child.tag) == name]


def _decode_uncompressed_binary(payload: bytes, header_dtype: np.dtype) -> bytes:
    header_nbytes = header_dtype.itemsize
    if len(payload) < header_nbytes:
        raise ValueError("Binary VTK payload is shorter than its length header")
    nbytes = int(np.frombuffer(payload[:header_nbytes], dtype=header_dtype, count=1)[0])
    data = payload[header_nbytes:header_nbytes + nbytes]
    if len(data) != nbytes:
        raise ValueError(f"Expected {nbytes} bytes from VTK payload, got {len(data)}")
    return data


def _parse_zlib_header(payload: bytes, header_dtype: np.dtype) -> tuple[int, int, np.ndarray]:
    if len(payload) < 3 * header_dtype.itemsize:
        raise ValueError("Compressed VTK payload header is incomplete")

    base_header = np.frombuffer(payload[: 3 * header_dtype.itemsize], dtype=header_dtype, count=3)
    n_blocks = int(base_header[0])
    comp_sizes_offset = 3 * header_dtype.itemsize
    comp_sizes_nbytes = n_blocks * header_dtype.itemsize
    header_nbytes = comp_sizes_offset + comp_sizes_nbytes
    if len(payload) < header_nbytes:
        raise ValueError("Compressed VTK payload is missing per-block sizes")

    comp_sizes = np.frombuffer(
        payload[comp_sizes_offset:header_nbytes],
        dtype=header_dtype,
        count=n_blocks,
    )
    return n_blocks, header_nbytes, comp_sizes


def _split_header_segment(text: str, header_dtype: np.dtype) -> tuple[bytes, str]:
    first_pad = text.find("=")
    if first_pad != -1:
        segment_end = first_pad + 1
        while segment_end < len(text) and text[segment_end] == "=":
            segment_end += 1
        if segment_end % 4 != 0:
            segment_end += 4 - (segment_end % 4)
        header_bytes = base64.b64decode(text[:segment_end], validate=False)
        _, header_nbytes, _ = _parse_zlib_header(header_bytes, header_dtype)
        if len(header_bytes) == header_nbytes:
            return header_bytes, text[segment_end:]

    for segment_end in range(4, min(len(text), 65536) + 1, 4):
        segment = text[:segment_end]
        try:
            header_bytes = base64.b64decode(segment, validate=True)
        except binascii.Error:
            continue
        try:
            _, header_nbytes, _ = _parse_zlib_header(header_bytes, header_dtype)
        except ValueError:
            continue
        if len(header_bytes) == header_nbytes:
            return header_bytes, text[segment_end:]
    raise ValueError("Could not isolate the VTK zlib header segment")


def _decode_zlib_binary_from_text(text: str, header_dtype: np.dtype) -> bytes:
    header_bytes, remaining = _split_header_segment(text, header_dtype)
    _, _, comp_sizes = _parse_zlib_header(header_bytes, header_dtype)
    compressed_payload = base64.b64decode(remaining, validate=False)

    chunks: list[bytes] = []
    cursor = 0
    for comp_size in comp_sizes:
        size = int(comp_size)
        block = compressed_payload[cursor:cursor + size]
        if len(block) != size:
            raise ValueError("Compressed VTK payload ended before all blocks were read")
        chunks.append(zlib.decompress(block))
        cursor += size
    return b"".join(chunks)


def _decode_zlib_binary(payload: bytes, header_dtype: np.dtype) -> bytes:
    _, header_nbytes, comp_sizes = _parse_zlib_header(payload, header_dtype)

    chunks: list[bytes] = []
    cursor = header_nbytes
    for comp_size in comp_sizes:
        size = int(comp_size)
        block = payload[cursor:cursor + size]
        if len(block) != size:
            raise ValueError("Compressed VTK payload is truncated inside a zlib block")
        chunks.append(zlib.decompress(block))
        cursor += size
    return b"".join(chunks)


def _decode_binary_payload(text: str, header_type: str, compressor: str | None) -> bytes:
    header_dtype = HEADER_TO_NUMPY.get(header_type, HEADER_TO_NUMPY["UInt32"])
    if compressor is None:
        payload = base64.b64decode("".join(text.split()))
        return _decode_uncompressed_binary(payload, header_dtype)
    if compressor != "vtkZLibDataCompressor":
        raise NotImplementedError(f"Unsupported VTK compressor: {compressor}")
    try:
        payload = base64.b64decode("".join(text.split()))
        return _decode_zlib_binary(payload, header_dtype)
    except ValueError:
        return _decode_zlib_binary_from_text("".join(text.split()), header_dtype)


def _decode_data_array(
    array_node: ET.Element,
    header_type: str,
    compressor: str | None,
) -> np.ndarray:
    vtk_type = array_node.attrib["type"]
    dtype = VTK_TO_NUMPY.get(vtk_type)
    if dtype is None:
        raise NotImplementedError(f"Unsupported VTK dtype: {vtk_type}")

    n_components = int(array_node.attrib.get("NumberOfComponents", "1"))
    fmt = array_node.attrib.get("format", "ascii")
    # VTK sometimes nests `InformationKey` metadata inside a `DataArray`. The
    # encoded payload lives in the node's direct text, not in the nested text.
    text = (array_node.text or "").strip()

    if fmt == "ascii":
        raw = np.fromstring(text, sep=" ", dtype=dtype)
    elif fmt == "binary":
        raw_bytes = _decode_binary_payload(text, header_type=header_type, compressor=compressor)
        raw = np.frombuffer(raw_bytes, dtype=dtype)
    elif fmt == "appended":
        raise NotImplementedError("Appended VTK arrays are not supported in this loader")
    else:
        raise ValueError(f"Unknown VTK array format: {fmt}")

    if n_components > 1:
        if raw.size % n_components != 0:
            raise ValueError(
                f"VTK array {array_node.attrib.get('Name', '<unnamed>')} has {raw.size} values "
                f"which is not divisible by NumberOfComponents={n_components}"
            )
        return raw.reshape(-1, n_components)
    return raw


def _load_named_arrays(
    parent: ET.Element | None,
    names: set[str] | None,
    header_type: str,
    compressor: str | None,
) -> dict[str, np.ndarray]:
    if parent is None:
        return {}

    arrays: dict[str, np.ndarray] = {}
    for array_node in _iter_children(parent, "DataArray"):
        name = array_node.attrib.get("Name")
        if not name:
            continue
        if names is not None and name not in names:
            continue
        arrays[name] = _decode_data_array(array_node, header_type=header_type, compressor=compressor)
    return arrays


def read_vtk_xml(
    path: str | Path,
    point_arrays: list[str] | tuple[str, ...] | None = None,
    cell_arrays: list[str] | tuple[str, ...] | None = None,
) -> VTKPiece:
    """Read points plus selected PointData/CellData arrays from a VTK XML file."""

    tree = ET.parse(path)
    root = tree.getroot()
    if _local_name(root.tag) != "VTKFile":
        raise ValueError(f"{path} is not a VTK XML file")

    header_type = root.attrib.get("header_type", "UInt32")
    compressor = root.attrib.get("compressor")

    dataset_node = None
    for child in root:
        if _local_name(child.tag) in {"UnstructuredGrid", "PolyData"}:
            dataset_node = child
            break
    if dataset_node is None:
        raise ValueError(f"{path} does not contain a supported VTK dataset node")

    piece = _find_first_child(dataset_node, "Piece")
    if piece is None:
        raise ValueError(f"{path} does not contain a Piece node")

    points_parent = _find_first_child(piece, "Points")
    if points_parent is None:
        raise ValueError(f"{path} does not contain point coordinates")

    point_array_nodes = _iter_children(points_parent, "DataArray")
    if len(point_array_nodes) != 1:
        raise ValueError(f"{path} expected exactly one Points/DataArray, found {len(point_array_nodes)}")
    points = _decode_data_array(point_array_nodes[0], header_type=header_type, compressor=compressor)
    if points.ndim != 2 or points.shape[1] not in {2, 3}:
        raise ValueError(f"{path} returned malformed point coordinates with shape {points.shape}")

    point_data = _load_named_arrays(
        _find_first_child(piece, "PointData"),
        names=set(point_arrays) if point_arrays is not None else None,
        header_type=header_type,
        compressor=compressor,
    )
    cell_data = _load_named_arrays(
        _find_first_child(piece, "CellData"),
        names=set(cell_arrays) if cell_arrays is not None else None,
        header_type=header_type,
        compressor=compressor,
    )

    return VTKPiece(points=np.asarray(points, dtype=np.float32), point_data=point_data, cell_data=cell_data)
