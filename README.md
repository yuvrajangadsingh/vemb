# vemb

httpie for embeddings. Embed text, images, audio, video, and PDFs from the command line.

```bash
pipx install vemb
export GEMINI_API_KEY=your_key
vemb text "hello world"
```

Powered by Gemini Embedding 2, the first natively multimodal embedding model.

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
vemb image photo.jpg                       # embed image (PNG, JPEG)
vemb audio clip.mp3                        # embed audio (MP3, WAV)
vemb video clip.mp4                        # embed video (MP4, MOV)
vemb pdf doc.pdf                           # embed PDF
vemb similar photo1.jpg photo2.jpg         # cosine similarity between two files
vemb search ./photos "sunset at beach"     # search a directory
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
vemb text "hello" --compact          # just the vector array
vemb text "hello" --numpy            # numpy format
vemb text "hello" --dim 768          # set dimensions (128-3072)
vemb text "hello" --task RETRIEVAL_QUERY   # set task type
```

## Search

Search indexes a directory and finds files similar to your query:

```bash
vemb search ./photos "sunset at beach" --top 5
```

Embeddings are cached in `.vemb/cache.json` inside the searched directory. Unchanged files won't be re-embedded on subsequent searches.

## License

MIT
