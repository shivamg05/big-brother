# Gemini Integration

This project supports a Gemini-backed extractor via `GeminiExtractor` in `src/big_brother/extractor.py`.

## Setup

1. Install dependencies:
   - `uv sync`
2. Set API key:
   - create `.env` in repo root with:
     - `GEMINI_API_KEY=your-key`
   - alternatively, use shell env:
     - `export GEMINI_API_KEY="your-key"`
     - or `export GOOGLE_API_KEY="your-key"`

## Usage

```python
from big_brother.extractor import GeminiExtractor
from big_brother.pipeline import WorkerMemoryPipeline

pipeline = WorkerMemoryPipeline(extractor=GeminiExtractor(model="gemini-2.5-flash"))
```

The extractor calls `client.models.generate_content(...)` with:
- `model="gemini-2.5-flash"`
- `config={"response_mime_type": "application/json", "response_json_schema": ..., "temperature": 0.1}`
- the sampled window images are sent as multimodal image parts (JPEG bytes), in order.
- built-in throttling and retry for quota limits (`requests_per_minute`, `max_retries`).

The model returns strict JSON mapped into `SubtaskEvent`.
