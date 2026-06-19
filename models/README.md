# Model artifacts

Large binaries (ONNX, vocab, trained weights) are **not** committed to this repo.

Generate them locally before running the proxy or Docker:

```bash
python -m src.download_vocab
python export_onnx.py          # or optimum-cli export
python -m src.quantize_onnx
python -m src.export_weights
```

Expected files in this directory:

- `vocab.txt`
- `all-MiniLM-L6-v2-int8.onnx`
- `router_weights.json`
