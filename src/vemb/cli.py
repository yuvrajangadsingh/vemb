import json
import sys
from pathlib import Path

import click

from vemb import __version__
from vemb.embed import (
    embed_text,
    embed_file,
    cosine_similarity,
    scan_supported_files,
    load_cache,
    save_cache,
    cache_key,
    MODEL,
)

GREEN = "\033[32m"
YELLOW = "\033[33m"
RED = "\033[31m"
BOLD = "\033[1m"
RESET = "\033[0m"


def format_result(embedding, compact=False, numpy_fmt=False):
    values = list(embedding.values)
    if compact:
        return json.dumps(values)
    if numpy_fmt:
        return f"np.array({values})"
    return json.dumps({
        "model": MODEL,
        "dimensions": len(values),
        "values": values,
    })


def _call_embed_text(text, **kwargs):
    try:
        return embed_text(text, **kwargs)
    except Exception as e:
        raise click.ClickException(str(e))


def _call_embed_file(filepath, **kwargs):
    try:
        return embed_file(filepath, **kwargs)
    except Exception as e:
        raise click.ClickException(str(e))


def embed_options(f):
    f = click.option("--dim", type=click.IntRange(128, 3072), help="Output dimensions (128-3072)")(f)
    f = click.option("--task", "task_type", help="Task type (e.g. RETRIEVAL_QUERY, SEMANTIC_SIMILARITY)")(f)
    f = click.option("--compact", is_flag=True, help="Output just the vector array")(f)
    f = click.option("--numpy", "numpy_fmt", is_flag=True, help="Output in numpy format")(f)
    return f


@click.group()
@click.version_option(version=__version__)
def cli():
    """httpie for embeddings. Embed anything from the command line."""
    pass


@cli.command()
@click.argument("content", default="-")
@embed_options
def text(content, dim, task_type, compact, numpy_fmt):
    """Embed text. Use - or pipe stdin."""
    if content == "-":
        if sys.stdin.isatty():
            raise click.ClickException("no text provided. Pass text or pipe stdin.")
        content = sys.stdin.read()
    if not content:
        raise click.ClickException("empty input")
    result = _call_embed_text(content, dim=dim, task_type=task_type)
    click.echo(format_result(result, compact=compact, numpy_fmt=numpy_fmt))


@cli.command()
@click.argument("files", nargs=-1, required=True, type=click.Path(exists=True, dir_okay=False))
@embed_options
@click.option("--jsonl", is_flag=True, help="Output one JSON object per line")
def embed(files, dim, task_type, compact, numpy_fmt, jsonl):
    """Embed one or more files (auto-detects type)."""
    for f in files:
        result = _call_embed_file(f, dim=dim, task_type=task_type)
        if jsonl:
            values = list(result.values)
            click.echo(json.dumps({"file": f, "model": MODEL, "dimensions": len(values), "values": values}))
        else:
            if len(files) > 1:
                click.echo(f"# {f}", err=True)
            click.echo(format_result(result, compact=compact, numpy_fmt=numpy_fmt))


@cli.command()
@click.argument("file", type=click.Path(exists=True, dir_okay=False))
@embed_options
def image(file, dim, task_type, compact, numpy_fmt):
    """Embed an image (PNG, JPEG)."""
    result = _call_embed_file(file, dim=dim, task_type=task_type)
    click.echo(format_result(result, compact=compact, numpy_fmt=numpy_fmt))


@cli.command()
@click.argument("file", type=click.Path(exists=True, dir_okay=False))
@embed_options
def audio(file, dim, task_type, compact, numpy_fmt):
    """Embed audio (MP3, WAV)."""
    result = _call_embed_file(file, dim=dim, task_type=task_type)
    click.echo(format_result(result, compact=compact, numpy_fmt=numpy_fmt))


@cli.command()
@click.argument("file", type=click.Path(exists=True, dir_okay=False))
@embed_options
def video(file, dim, task_type, compact, numpy_fmt):
    """Embed video (MP4, MOV)."""
    result = _call_embed_file(file, dim=dim, task_type=task_type)
    click.echo(format_result(result, compact=compact, numpy_fmt=numpy_fmt))


@cli.command()
@click.argument("file", type=click.Path(exists=True, dir_okay=False))
@embed_options
def pdf(file, dim, task_type, compact, numpy_fmt):
    """Embed a PDF document."""
    result = _call_embed_file(file, dim=dim, task_type=task_type)
    click.echo(format_result(result, compact=compact, numpy_fmt=numpy_fmt))


@cli.command()
@click.argument("file1", type=click.Path(exists=True, dir_okay=False))
@click.argument("file2", type=click.Path(exists=True, dir_okay=False))
@click.option("--dim", type=click.IntRange(128, 3072), help="Output dimensions (128-3072)")
def similar(file1, file2, dim):
    """Compute cosine similarity between two files."""
    a = _call_embed_file(file1, dim=dim)
    b = _call_embed_file(file2, dim=dim)
    score = cosine_similarity(a.values, b.values)
    if score >= 0.8:
        color = GREEN
    elif score >= 0.5:
        color = YELLOW
    else:
        color = RED
    click.echo(f"{color}{BOLD}{score:.4f}{RESET}  {file1} <-> {file2}")


@cli.command()
@click.argument("directory", type=click.Path(exists=True, file_okay=False))
@click.argument("query")
@click.option("--dim", type=click.IntRange(128, 3072), help="Output dimensions (128-3072)")
@click.option("--top", default=10, type=click.IntRange(min=1), help="Number of results to show")
@click.option("--no-cache", is_flag=True, help="Skip cache")
def search(directory, query, dim, top, no_cache):
    """Search a directory for files similar to a query."""
    import numpy as np

    files = scan_supported_files(directory)
    if not files:
        raise click.ClickException("no supported files found in directory")

    base = Path(directory)
    keys, cache_matrix = ({}, None) if no_cache else load_cache(directory, dim)

    query_emb = _call_embed_text(query, dim=dim, task_type="RETRIEVAL_QUERY")
    query_vec = np.asarray(query_emb.values, dtype=np.float32)
    qnorm = float(np.linalg.norm(query_vec))
    if qnorm > 0:
        query_vec = query_vec / qnorm

    D = int(cache_matrix.shape[1]) if cache_matrix is not None else query_vec.shape[0]
    working = np.empty((len(files), D), dtype=np.float32)
    new_keys = dict(keys)
    new_rows = []
    uncached = sum(1 for f in files if cache_key(f, base) not in keys)

    embedded = 0
    for i, f in enumerate(files):
        k = cache_key(f, base)
        if k in keys and cache_matrix is not None:
            working[i] = cache_matrix[keys[k]]
            continue
        embedded += 1
        print(f"  Embedding {embedded}/{uncached}...", end="\r", file=sys.stderr)
        emb = _call_embed_file(str(f), dim=dim, task_type="RETRIEVAL_DOCUMENT")
        vec = np.asarray(emb.values, dtype=np.float32)
        vnorm = float(np.linalg.norm(vec))
        if vnorm > 0:
            vec = vec / vnorm
        working[i] = vec
        base_rows = cache_matrix.shape[0] if cache_matrix is not None else 0
        new_keys[k] = base_rows + len(new_rows)
        new_rows.append(vec)

    if embedded:
        print(" " * 40, end="\r", file=sys.stderr)

    if not no_cache and new_rows:
        new_block = np.stack(new_rows).astype(np.float32, copy=False)
        updated = new_block if cache_matrix is None else np.vstack([cache_matrix, new_block])
        save_cache(directory, dim, new_keys, updated)

    scores = (working @ query_vec).tolist()
    results = sorted(zip(scores, files), key=lambda x: x[0], reverse=True)
    for score, f in results[:top]:
        if score >= 0.8:
            color = GREEN
        elif score >= 0.5:
            color = YELLOW
        else:
            color = RED
        click.echo(f"  {color}{score:.3f}{RESET}  {f}")
