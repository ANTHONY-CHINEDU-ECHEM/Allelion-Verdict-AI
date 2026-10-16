# Deployment

## Rebuild everything from source

```bash
pip install -e ".[dev]"
make all              # acquire, knowledge base, evidence, models, literature, graph, tuning, indexes, verifier, development evaluation, bundle
make evaluate_blind   # once, after every component is frozen; writes reports/ and figures
make web              # React interface
make test             # Python tests and the browser parity tests
make serve            # FastAPI on port 8000, also serving web/dist
```

Everything trains on CPU. Most of the build time goes to threshold tuning on the development set and reranker training. `make postmortem` slices the stored blind predictions without scoring them again.

## Docker

```bash
make all && make web
docker compose up --build                    # service and interface on http://localhost:8000
docker compose --profile llm up --build      # adds Ollama; then set ALLELION_LLM=ollama
```

`data/` and `reports/` are mounted, so the image stays small and the same build artefacts serve both.

## Static deployment

The interface needs no server: copy `web/dist` (which contains `bundle/`) to any static host. The bundle is about 15 MB of gzip compressed JSON and one file of half precision word vectors, decompressed in the browser with `DecompressionStream`; the search index is built in the browser the first time the copilot opens.

Some hosts serve only web and text file types and refuse `.gz` files. `make artifact` writes a copy to `build/artifact` in which every gzip file is base64 encoded as `<name>.b64.txt` and the bundle manifest is marked `"encoding": "gzip+base64"`; the loader reads that flag and decodes before decompressing. The live demo is published this way.

## Inside a hospital network

Every model trains on CPU from public data, the copilot's default writer runs without a language model, and the optional writer uses a local Ollama server. No request leaves the network. The audit log stores redacted questions and hashes only.
