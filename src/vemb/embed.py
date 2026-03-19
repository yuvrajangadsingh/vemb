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
    if dim:
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


def cache_key(filepath):
    p = Path(filepath)
    stat = p.stat()
    return f"{p.name}:{stat.st_size}:{stat.st_mtime_ns}"


def load_cache(directory, dim):
    cache_path = Path(directory) / ".vemb" / "cache.json"
    if not cache_path.exists():
        return {}
    try:
        data = json.loads(cache_path.read_text())
        if data.get("model") != MODEL or data.get("dim") != dim:
            return {}
        return data.get("entries", {})
    except (json.JSONDecodeError, KeyError):
        return {}


def save_cache(directory, dim, entries):
    cache_dir = Path(directory) / ".vemb"
    cache_dir.mkdir(exist_ok=True)
    data = {"version": 1, "model": MODEL, "dim": dim, "entries": entries}
    (cache_dir / "cache.json").write_text(json.dumps(data))


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
