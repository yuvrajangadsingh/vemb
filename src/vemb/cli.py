import json
import sys

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
DIM = "\033[2m"
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


def embed_options(f):
    f = click.option("--dim", type=int, help="Output dimensions (128-3072)")(f)
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
@click.argument("content")
@embed_options
def text(content, dim, task_type, compact, numpy_fmt):
    """Embed text."""
    result = embed_text(content, dim=dim, task_type=task_type)
    click.echo(format_result(result, compact=compact, numpy_fmt=numpy_fmt))


@cli.command()
@click.argument("file", type=click.Path(exists=True))
@embed_options
def image(file, dim, task_type, compact, numpy_fmt):
    """Embed an image (PNG, JPEG)."""
    result = embed_file(file, dim=dim, task_type=task_type)
    click.echo(format_result(result, compact=compact, numpy_fmt=numpy_fmt))


@cli.command()
@click.argument("file", type=click.Path(exists=True))
@embed_options
def audio(file, dim, task_type, compact, numpy_fmt):
    """Embed audio (MP3, WAV)."""
    result = embed_file(file, dim=dim, task_type=task_type)
    click.echo(format_result(result, compact=compact, numpy_fmt=numpy_fmt))


@cli.command()
@click.argument("file", type=click.Path(exists=True))
@embed_options
def video(file, dim, task_type, compact, numpy_fmt):
    """Embed video (MP4, MOV)."""
    result = embed_file(file, dim=dim, task_type=task_type)
    click.echo(format_result(result, compact=compact, numpy_fmt=numpy_fmt))


@cli.command()
@click.argument("file", type=click.Path(exists=True))
@embed_options
def pdf(file, dim, task_type, compact, numpy_fmt):
    """Embed a PDF document."""
    result = embed_file(file, dim=dim, task_type=task_type)
    click.echo(format_result(result, compact=compact, numpy_fmt=numpy_fmt))


@cli.command()
@click.argument("file1", type=click.Path(exists=True))
@click.argument("file2", type=click.Path(exists=True))
@click.option("--dim", type=int, help="Output dimensions (128-3072)")
def similar(file1, file2, dim):
    """Compute cosine similarity between two files."""
    a = embed_file(file1, dim=dim)
    b = embed_file(file2, dim=dim)
    score = cosine_similarity(a.values, b.values)
    if score >= 0.8:
        color = GREEN
    elif score >= 0.5:
        color = YELLOW
    else:
        color = RED
    click.echo(f"{color}{BOLD}{score:.4f}{RESET}  {file1} <-> {file2}")


@cli.command()
@click.argument("directory", type=click.Path(exists=True))
@click.argument("query")
@click.option("--dim", type=int, help="Output dimensions (128-3072)")
@click.option("--top", default=10, help="Number of results to show")
@click.option("--no-cache", is_flag=True, help="Skip cache")
def search(directory, query, dim, top, no_cache):
    """Search a directory for files similar to a query."""
    files = scan_supported_files(directory)
    if not files:
        click.echo("No supported files found.", err=True)
        sys.exit(1)

    cache = {} if no_cache else load_cache(directory, dim)
    query_emb = embed_text(query, dim=dim, task_type="RETRIEVAL_QUERY")

    results = []
    new_cache = dict(cache)
    uncached = 0
    for f in files:
        key = cache_key(f)
        if key not in cache:
            uncached += 1

    embedded = 0
    for f in files:
        key = cache_key(f)
        if key in cache:
            values = cache[key]["values"]
        else:
            embedded += 1
            print(f"  Embedding {embedded}/{uncached}...", end="\r", file=sys.stderr)
            emb = embed_file(str(f), dim=dim, task_type="RETRIEVAL_DOCUMENT")
            values = list(emb.values)
            new_cache[key] = {"file": str(f), "values": values}
        score = cosine_similarity(query_emb.values, values)
        results.append((score, f))

    if embedded:
        print(" " * 40, end="\r", file=sys.stderr)

    if not no_cache:
        save_cache(directory, dim, new_cache)

    results.sort(key=lambda x: x[0], reverse=True)
    for score, f in results[:top]:
        if score >= 0.8:
            color = GREEN
        elif score >= 0.5:
            color = YELLOW
        else:
            color = RED
        click.echo(f"  {color}{score:.3f}{RESET}  {f}")
