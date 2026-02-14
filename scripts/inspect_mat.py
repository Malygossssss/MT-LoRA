#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
inspect_mat.py - Inspect MATLAB .mat file contents (v7.2 and v7.3/HDF5).

Usage:
  python inspect_mat.py /path/to/file.mat
  python inspect_mat.py file.mat --max-depth 6 --max-preview 12
"""

import argparse
import sys
from typing import Any

import numpy as np

# Optional imports (we'll import lazily where needed)
# scipy.io for v7.2 and earlier
# h5py for v7.3


# -------------------------
# Pretty-print helpers
# -------------------------
def _indent(level: int) -> str:
    return "  " * level


def _safe_preview_array(a: np.ndarray, max_preview: int) -> str:
    try:
        flat = a.ravel()
        n = flat.size
        take = min(n, max_preview)
        sample = flat[:take]
        # Convert to Python scalars where possible for clean printing
        sample_list = [x.item() if hasattr(x, "item") else x for x in sample]
        suffix = " ..." if n > take else ""
        return f"{sample_list}{suffix}"
    except Exception as e:
        return f"<preview failed: {e}>"


def _summarize_numpy(a: np.ndarray, level: int, max_preview: int) -> None:
    print(
        f"{_indent(level)}ndarray shape={a.shape} dtype={a.dtype} "
        f"min={np.nanmin(a) if a.size else 'NA'} max={np.nanmax(a) if a.size else 'NA'}"
        if a.size and np.issubdtype(a.dtype, np.number)
        else f"{_indent(level)}ndarray shape={a.shape} dtype={a.dtype}"
    )
    if a.size:
        print(f"{_indent(level)}preview: {_safe_preview_array(a, max_preview)}")


def _is_mat_struct(obj: Any) -> bool:
    # scipy.io.matlab.mio5_params.mat_struct (old) or scipy.io.matlab._mio5_params.mat_struct (new)
    return obj.__class__.__name__ == "mat_struct"


def _summarize_obj(
    name: str,
    obj: Any,
    level: int,
    max_depth: int,
    max_preview: int,
) -> None:
    if level > max_depth:
        print(f"{_indent(level)}{name}: <max depth reached>")
        return

    prefix = f"{_indent(level)}{name}: "

    # None / scalar
    if obj is None:
        print(prefix + "None")
        return

    # Numpy array
    if isinstance(obj, np.ndarray):
        print(prefix + f"ndarray (ndim={obj.ndim})")
        _summarize_numpy(obj, level + 1, max_preview)
        return

    # Dict
    if isinstance(obj, dict):
        print(prefix + f"dict keys={list(obj.keys())}")
        for k, v in obj.items():
            _summarize_obj(str(k), v, level + 1, max_depth, max_preview)
        return

    # List / tuple
    if isinstance(obj, (list, tuple)):
        print(prefix + f"{type(obj).__name__} len={len(obj)}")
        for i, v in enumerate(obj[: max_preview]):
            _summarize_obj(f"[{i}]", v, level + 1, max_depth, max_preview)
        if len(obj) > max_preview:
            print(f"{_indent(level+1)}... ({len(obj) - max_preview} more items)")
        return

    # scipy mat_struct
    if _is_mat_struct(obj):
        fields = getattr(obj, "_fieldnames", []) or []
        print(prefix + f"mat_struct fields={fields}")
        for f in fields:
            try:
                _summarize_obj(f, getattr(obj, f), level + 1, max_depth, max_preview)
            except Exception as e:
                print(f"{_indent(level+1)}{f}: <read failed: {e}>")
        return

    # Basic python scalar / string
    if isinstance(obj, (str, int, float, bool, complex, bytes)):
        v = obj if not isinstance(obj, bytes) else obj[:max_preview]
        print(prefix + f"{type(obj).__name__} value={v}")
        return

    # Fallback
    print(prefix + f"{type(obj).__name__} (unhandled)")
    try:
        s = str(obj)
        if len(s) > 200:
            s = s[:200] + "..."
        print(f"{_indent(level+1)}str: {s}")
    except Exception:
        pass


# -------------------------
# MATLAB v7.2 and earlier
# -------------------------
def inspect_mat_v72(path: str, max_depth: int, max_preview: int) -> None:
    from scipy.io import loadmat

    md = loadmat(path, struct_as_record=False, squeeze_me=True)
    # Remove MATLAB metadata keys for clarity
    meta_keys = {"__header__", "__version__", "__globals__"}
    keys = [k for k in md.keys() if k not in meta_keys]

    print(f"[v7.2-or-earlier] {path}")
    print(f"Variables: {keys}\n")

    for k in keys:
        _summarize_obj(k, md[k], level=0, max_depth=max_depth, max_preview=max_preview)
        print()


# -------------------------
# MATLAB v7.3 (HDF5)
# -------------------------
def inspect_mat_v73(path: str, max_depth: int, max_preview: int) -> None:
    import h5py

    def walk(name: str, obj: Any, level: int) -> None:
        if level > max_depth:
            print(f"{_indent(level)}{name}: <max depth reached>")
            return

        if isinstance(obj, h5py.Group):
            keys = list(obj.keys())
            print(f"{_indent(level)}{name}/ (Group) keys={keys}")
            for k in keys:
                walk(k, obj[k], level + 1)

        elif isinstance(obj, h5py.Dataset):
            shape = obj.shape
            dtype = obj.dtype
            print(f"{_indent(level)}{name}: (Dataset) shape={shape} dtype={dtype}")
            try:
                # Avoid loading huge arrays fully
                if obj.size == 0:
                    return
                # Read a small slice/flattened preview
                data = obj[()]
                # data might be numpy scalar / array
                if isinstance(data, np.ndarray):
                    if data.size:
                        print(f"{_indent(level+1)}preview: {_safe_preview_array(data, max_preview)}")
                else:
                    print(f"{_indent(level+1)}value: {data}")
            except Exception as e:
                print(f"{_indent(level+1)}<read failed: {e}>")
        else:
            print(f"{_indent(level)}{name}: {type(obj).__name__}")

    print(f"[v7.3/HDF5] {path}\n")
    with h5py.File(path, "r") as f:
        for k in f.keys():
            walk(k, f[k], level=0)
            print()


# -------------------------
# Version detection + CLI
# -------------------------
def is_hdf5_mat(path: str) -> bool:
    try:
        import h5py

        return h5py.is_hdf5(path)
    except Exception:
        return False


def main():
    ap = argparse.ArgumentParser(description="Inspect MATLAB .mat contents (v7.2 and v7.3).")
    ap.add_argument("mat_path", help="Path to .mat file")
    ap.add_argument("--max-depth", type=int, default=5, help="Max recursion depth (default: 5)")
    ap.add_argument("--max-preview", type=int, default=10, help="Max preview items for arrays/lists (default: 10)")
    args = ap.parse_args()

    path = args.mat_path

    try:
        if is_hdf5_mat(path):
            inspect_mat_v73(path, max_depth=args.max_depth, max_preview=args.max_preview)
        else:
            inspect_mat_v72(path, max_depth=args.max_depth, max_preview=args.max_preview)
    except ImportError as e:
        print(f"Missing dependency: {e}", file=sys.stderr)
        print("Install with: pip install scipy h5py numpy", file=sys.stderr)
        sys.exit(2)
    except FileNotFoundError:
        print(f"File not found: {path}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
