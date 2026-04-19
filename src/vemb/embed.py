import json
import math
import mimetypes
import os
import sys
from pathlib import Path

from google import genai
from google.genai import types

MODEL = "gemini-embedding-2-preview"

SUPPORTED_MIMES = {
    "image/png",
    "image/jpeg",
    "audio/mpeg",
    "audio/x-wav",
    "audio/wav",
    "video/mp4",
    "video/quicktime",
    "application/pdf",
}


def get_client():
    key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not key:
        print("Error: set GEMINI_API_KEY or GOOGLE_API_KEY env var", file=sys.stderr)
        sys.exit(1)
    return genai.Client(api_key=key)


def guess_mime(filepath):
    mime, _ = mimetypes.guess_type(filepath)
    if not mime or mime not in SUPPORTED_MIMES:
        supported = ", ".join(sorted(SUPPORTED_MIMES))
        print(f"Error: unsupported file type '{mime or 'unknown'}' for {filepath}", file=sys.stderr)
        print(f"Supported: {supported}", file=sys.stderr)
        sys.exit(1)
    return mime


def _build_config(dim=None, task_type=None):
    kwargs = {}
    if dim is not None:
        kwargs["output_dimensionality"] = dim
    if task_type:
        kwargs["task_type"] = task_type
    return types.EmbedContentConfig(**kwargs) if kwargs else None


def embed_text(text, dim=None, task_type=None):
    client = get_client()
    result = client.models.embed_content(
        model=MODEL,
        contents=text,
        config=_build_config(dim, task_type),
    )
    return result.embeddings[0]


def embed_file(filepath, dim=None, task_type=None):
    mime = guess_mime(filepath)
    data = Path(filepath).read_bytes()
    client = get_client()
    result = client.models.embed_content(
        model=MODEL,
        contents=types.Content(
            parts=[types.Part.from_bytes(data=data, mime_type=mime)]
        ),
        config=_build_config(dim, task_type),
    )
    return result.embeddings[0]


def cosine_similarity(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def cache_key(filepath, base_dir=None):
    p = Path(filepath)
    stat = p.stat()
    rel = str(p.relative_to(base_dir)) if base_dir else str(p)
    return f"{rel}:{stat.st_size}:{stat.st_mtime_ns}"


CACHE_VERSION = 2


def _cache_paths(directory):
    cache_dir = Path(directory) / ".vemb"
    return cache_dir, cache_dir / "manifest.json", cache_dir / "vectors.npy"


def _normalize_rows_inplace(matrix):
    import numpy as np
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    np.divide(matrix, norms, out=matrix)


def _migrate_legacy_cache(directory, dim):
    cache_dir, manifest_path, vectors_path = _cache_paths(directory)
    legacy = cache_dir / "cache.json"
    if manifest_path.exists() or not legacy.exists():
        return
    try:
        data = json.loads(legacy.read_text())
    except (json.JSONDecodeError, OSError):
        return
    if data.get("model") != MODEL or data.get("dim") != dim:
        return
    entries = data.get("entries") or {}
    if not entries:
        return
    import numpy as np
    first_values = next(iter(entries.values()))["values"]
    D = len(first_values)
    N = len(entries)
    matrix = np.empty((N, D), dtype=np.float32)
    keys = {}
    for i, (k, v) in enumerate(entries.items()):
        matrix[i] = np.asarray(v["values"], dtype=np.float32)
        keys[k] = i
    _normalize_rows_inplace(matrix)
    cache_dir.mkdir(exist_ok=True)
    np.save(vectors_path, matrix, allow_pickle=False)
    manifest = {
        "version": CACHE_VERSION,
        "model": MODEL,
        "dim": dim,
        "normalized": True,
        "keys": keys,
    }
    manifest_path.write_text(json.dumps(manifest))
    legacy.unlink()
    print(f"  Migrated cache to binary format ({N} vectors, {D} dim).", file=sys.stderr)


def load_cache(directory, dim):
    _migrate_legacy_cache(directory, dim)
    cache_dir, manifest_path, vectors_path = _cache_paths(directory)
    if not manifest_path.exists() or not vectors_path.exists():
        return {}, None
    try:
        manifest = json.loads(manifest_path.read_text())
    except (json.JSONDecodeError, OSError):
        return {}, None
    if (
        manifest.get("model") != MODEL
        or manifest.get("dim") != dim
        or manifest.get("version") != CACHE_VERSION
    ):
        return {}, None
    import numpy as np
    try:
        matrix = np.load(vectors_path, mmap_mode=None, allow_pickle=False)
    except (ValueError, OSError):
        return {}, None
    if matrix.dtype != np.float32 or matrix.ndim != 2 or matrix.shape[1] != dim:
        return {}, None
    keys = manifest.get("keys") or {}
    return keys, matrix


def save_cache(directory, dim, keys, matrix):
    import numpy as np
    cache_dir, manifest_path, vectors_path = _cache_paths(directory)
    cache_dir.mkdir(exist_ok=True)
    if matrix.dtype != np.float32:
        matrix = matrix.astype(np.float32)
    tmp_vectors = vectors_path.with_name(vectors_path.name + ".tmp")
    with open(tmp_vectors, "wb") as f:
        np.save(f, matrix, allow_pickle=False)
    tmp_vectors.replace(vectors_path)
    manifest = {
        "version": CACHE_VERSION,
        "model": MODEL,
        "dim": dim,
        "normalized": True,
        "keys": keys,
    }
    tmp_manifest = manifest_path.with_suffix(".json.tmp")
    tmp_manifest.write_text(json.dumps(manifest))
    tmp_manifest.replace(manifest_path)


def scan_supported_files(directory):
    files = []
    for p in sorted(Path(directory).rglob("*")):
        if ".vemb" in p.parts:
            continue
        if p.is_file():
            mime, _ = mimetypes.guess_type(str(p))
            if mime and mime in SUPPORTED_MIMES:
                files.append(p)
    return files
