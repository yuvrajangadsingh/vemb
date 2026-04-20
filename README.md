# vemb

httpie for embeddings. Embed text, images, audio, video, and PDFs from the command line.

<p align="center">
  <img src="demo.gif" alt="vemb demo" width="700">
</p>

```bash
pipx install vemb
export GEMINI_API_KEY=your_key
vemb text "hello world"
```

Powered by [Gemini Embedding 2](https://blog.google/innovation-and-ai/models-and-research/gemini-models/gemini-embedding-2/), the first natively multimodal embedding model. One model, one vector space for everything.

## Install

```bash
pipx install vemb
# or
pip install vemb
```

Get a free API key at https://aistudio.google.com/apikey

```bash
export GEMINI_API_KEY=your_key
```

## Commands

```bash
vemb text "hello world"                    # embed text
vemb embed photo.jpg                       # embed any file (auto-detects type)
vemb embed *.jpg --jsonl                   # batch embed, one JSON per line
vemb image photo.jpg                       # embed image (PNG, JPEG)
vemb audio clip.mp3                        # embed audio (MP3, WAV)
vemb video clip.mp4                        # embed video (MP4, MOV)
vemb pdf doc.pdf                           # embed PDF
vemb similar photo1.jpg photo2.jpg         # cosine similarity between two files
vemb search ./photos "sunset at beach"     # search a directory
```

Pipe from stdin:

```bash
echo "hello world" | vemb text -
cat document.txt | vemb text -
```

## Output

Default output is JSON:

```json
{
  "model": "gemini-embedding-2-preview",
  "dimensions": 768,
  "values": [0.012, -0.034, ...]
}
```

Options:

```bash
vemb text "hello" --compact                # just the vector array
vemb text "hello" --numpy                  # numpy format
vemb text "hello" --dim 768                # set dimensions (128-3072)
vemb text "hello" --task RETRIEVAL_QUERY   # set task type
```

Batch mode outputs JSONL (one embedding per line):

```bash
vemb embed *.jpg --jsonl > embeddings.jsonl
```

## Search

Search indexes a directory and finds files similar to your query:

```bash
vemb search ./photos "sunset at beach" --top 5
```

Embeddings are cached as a binary `numpy` matrix in `.vemb/vectors.npy` with a lightweight `.vemb/manifest.json` mapping keys to row indices. Vectors are pre-normalized so cosine reduces to a dot product at query time. Unchanged files won't be re-embedded on subsequent searches.

Legacy `.vemb/cache.json` caches (from v0.2.0 and earlier) are migrated to the binary format automatically on first load.

## Supported formats

| Type | Formats |
|------|---------|
| Text | any string, stdin |
| Image | PNG, JPEG |
| Audio | MP3, WAV (up to 80s) |
| Video | MP4, MOV (up to 128s) |
| PDF | up to 6 pages |

## License

MIT
