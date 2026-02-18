#!/usr/bin/env python3
"""
One-shot reproducibility diagnosis for MT-LoRA PASCAL_MT runs.

Usage example:
  python scripts/repro_diagnose.py \
    --pascal-root /path/to/PASCAL_MT \
    --output reports/repro_diag_a100.json

Run on both servers and compare JSON outputs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple


DEFAULT_IDS = [
    "2008_005016",
    "2009_004797",
    "2008_006355",
    "2008_003209",
    "2009_002629",
    "2008_006960",
    "2008_006973",
    "2009_000636",
]


def run_cmd(cmd: List[str]) -> Dict[str, object]:
    try:
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False)
        return {
            "cmd": " ".join(cmd),
            "returncode": proc.returncode,
            "stdout": proc.stdout.strip(),
            "stderr": proc.stderr.strip(),
        }
    except Exception as exc:  # pragma: no cover
        return {"cmd": " ".join(cmd), "error": repr(exc)}


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> Optional[str]:
    if not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def safe_import_version(module_name: str) -> Dict[str, Optional[str]]:
    out = {"module": module_name, "version": None, "error": None}
    try:
        module = __import__(module_name)
        out["version"] = getattr(module, "__version__", "<no __version__>")
    except Exception as exc:
        out["error"] = repr(exc)
    return out


def infer_semseg_path(root: Path, image_id: str) -> Tuple[Optional[Path], str]:
    voc = root / "semseg" / "VOC12" / f"{image_id}.png"
    ctx = root / "semseg" / "pascal-context" / f"{image_id}.png"
    if voc.is_file():
        return voc, "VOC12"
    if ctx.is_file():
        return ctx, "pascal-context"
    return None, "missing"


def read_ids(ids_arg: Optional[str]) -> List[str]:
    if not ids_arg:
        return DEFAULT_IDS
    p = Path(ids_arg)
    if p.is_file():
        ids: List[str] = []
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                ids.append(line)
        return ids
    return [x.strip() for x in ids_arg.split(",") if x.strip()]


def tree_manifest_hash(root: Path) -> Dict[str, object]:
    if not root.exists():
        return {"exists": False}
    rel_paths: List[str] = []
    file_count = 0
    dir_count = 0
    for dpath, dnames, fnames in os.walk(root):
        dir_count += len(dnames)
        for fn in fnames:
            file_count += 1
            full = Path(dpath) / fn
            rel_paths.append(str(full.relative_to(root)))
    rel_paths.sort()
    h = hashlib.sha256()
    for rp in rel_paths:
        h.update(rp.encode("utf-8"))
        h.update(b"\n")
    return {
        "exists": True,
        "file_count": file_count,
        "dir_count": dir_count,
        "pathlist_sha256": h.hexdigest(),
    }


def filelist_content_hash(root: Path) -> Dict[str, object]:
    if not root.exists():
        return {"exists": False}
    h = hashlib.sha256()
    file_count = 0
    total_bytes = 0
    for dpath, _, fnames in os.walk(root):
        for fn in sorted(fnames):
            file_count += 1
            fp = Path(dpath) / fn
            rel = str(fp.relative_to(root))
            h.update(rel.encode("utf-8"))
            h.update(b"\0")
            st = fp.stat()
            total_bytes += st.st_size
            h.update(str(st.st_size).encode("utf-8"))
            h.update(b"\0")
            fh = sha256_file(fp)
            h.update((fh or "MISSING").encode("utf-8"))
            h.update(b"\n")
    return {
        "exists": True,
        "file_count": file_count,
        "total_bytes": total_bytes,
        "content_sha256": h.hexdigest(),
    }


def collect_env() -> Dict[str, object]:
    env_keys = [
        "CUDA_VISIBLE_DEVICES",
        "CUBLAS_WORKSPACE_CONFIG",
        "PYTHONHASHSEED",
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NCCL_DEBUG",
        "NCCL_P2P_DISABLE",
        "NCCL_IB_DISABLE",
    ]
    env = {k: os.environ.get(k) for k in env_keys}
    versions = [
        safe_import_version("torch"),
        safe_import_version("torchvision"),
        safe_import_version("numpy"),
        safe_import_version("cv2"),
        safe_import_version("PIL"),
        safe_import_version("scipy"),
        safe_import_version("timm"),
    ]

    torch_details: Dict[str, object] = {}
    try:
        import torch
        torch_details = {
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda,
            "cudnn_version": torch.backends.cudnn.version(),
            "device_count": torch.cuda.device_count(),
            "device_names": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
            "allow_tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
            "allow_tf32_cudnn": torch.backends.cudnn.allow_tf32,
            "float32_matmul_precision": torch.get_float32_matmul_precision(),
        }
    except Exception as exc:
        torch_details = {"error": repr(exc)}

    return {
        "host": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version,
        "env": env,
        "versions": versions,
        "torch_details": torch_details,
        "nvidia_smi": run_cmd(["nvidia-smi"]),
        "pip_freeze": run_cmd([sys.executable, "-m", "pip", "freeze"]),
    }


def collect_dataset(root: Path, ids: List[str], include_full_hash: bool) -> Dict[str, object]:
    key_dirs = [
        "JPEGImages",
        "semseg/VOC12",
        "semseg/pascal-context",
        "normals_distill",
        "sal_distill",
        "human_parts",
        "ImageSets/Context",
        "ImageSets/Parts",
    ]

    dir_stats = {}
    for kd in key_dirs:
        p = root / kd
        exists = p.exists()
        count = 0
        if exists and p.is_dir():
            count = sum(1 for _ in p.iterdir())
        dir_stats[kd] = {"exists": exists, "entries": count}

    split_files = [
        root / "ImageSets" / "Context" / "train.txt",
        root / "ImageSets" / "Context" / "val.txt",
        root / "ImageSets" / "Parts" / "trainval.txt",
    ]
    split_info = {}
    for sf in split_files:
        split_info[str(sf.relative_to(root))] = {
            "exists": sf.is_file(),
            "sha256": sha256_file(sf),
            "line_count": len(sf.read_text(encoding="utf-8").splitlines()) if sf.is_file() else None,
        }

    parts_cache = root / "ImageSets" / "Parts" / "trainval.txt"

    per_id = {}
    for image_id in ids:
        image_path = root / "JPEGImages" / f"{image_id}.jpg"
        semseg_path, semseg_source = infer_semseg_path(root, image_id)
        normals_path = root / "normals_distill" / f"{image_id}.png"
        sal_path = root / "sal_distill" / f"{image_id}.png"
        human_parts_path = root / "human_parts" / f"{image_id}.mat"

        per_id[image_id] = {
            "image": {"path": str(image_path), "exists": image_path.is_file(), "sha256": sha256_file(image_path)},
            "semseg": {
                "source": semseg_source,
                "path": str(semseg_path) if semseg_path else None,
                "exists": semseg_path.is_file() if semseg_path else False,
                "sha256": sha256_file(semseg_path) if semseg_path else None,
            },
            "normals_distill": {"path": str(normals_path), "exists": normals_path.is_file(), "sha256": sha256_file(normals_path)},
            "sal_distill": {"path": str(sal_path), "exists": sal_path.is_file(), "sha256": sha256_file(sal_path)},
            "human_parts": {"path": str(human_parts_path), "exists": human_parts_path.is_file(), "sha256": sha256_file(human_parts_path)},
        }

    out = {
        "root": str(root),
        "root_exists": root.exists(),
        "root_manifest": tree_manifest_hash(root),
        "key_dirs": dir_stats,
        "split_files": split_info,
        "parts_cache_trainval_txt": {"path": str(parts_cache), "exists": parts_cache.is_file(), "sha256": sha256_file(parts_cache)},
        "per_id": per_id,
    }

    if include_full_hash:
        out["full_content_hash"] = filelist_content_hash(root)

    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Diagnose cross-server reproducibility issues for MT-LoRA")
    parser.add_argument("--pascal-root", required=True, help="Path to PASCAL_MT root")
    parser.add_argument("--ids", default=None, help="Comma-separated ids OR a text file containing one id per line")
    parser.add_argument("--full-hash", action="store_true", help="Compute full content hash of the dataset root (slow)")
    parser.add_argument("--output", default=None, help="Output JSON path. Default: reports/repro_diag_<host>_<ts>.json")
    args = parser.parse_args()

    ids = read_ids(args.ids)
    pascal_root = Path(args.pascal_root).expanduser().resolve()

    report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
        "ids": ids,
        "env": collect_env(),
        "dataset": collect_dataset(pascal_root, ids, include_full_hash=args.full_hash),
        "how_to_compare": [
            "Run this script on both servers.",
            "Diff JSON files directly, focusing on env.versions, env.torch_details, dataset.split_files, dataset.per_id.",
            "Any mismatch in per_id hashes or semseg source implies data/label pipeline mismatch.",
        ],
    }

    output = args.output
    if not output:
        host = socket.gethostname().replace("/", "_")
        ts = time.strftime("%Y%m%d_%H%M%S", time.localtime())
        output = f"reports/repro_diag_{host}_{ts}.json"

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[ok] report written to: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
