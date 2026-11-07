"""Lädt Gesetzeskorpus und Modelle einmalig in den Lab-Ordner (braucht Netzzugang).

Aufruf im Ordner labs/:
    python prefetch.py            # alles
    python prefetch.py --check    # nur prüfen, was vorhanden ist

Danach laufen die Labs offline (außer dem Generator-LLM in Lab 5):
    labs/data/gesetze/normen.parquet   eine Zeile je Paragraph/Artikel
    labs/models/<name>/                Gewichte, Tokenizer, Konfiguration
Die Modell-Revisionen stehen nach dem ersten Lauf in models/revisions.json.
"""
from __future__ import annotations

import json
import sys

from huggingface_hub import HfApi, hf_hub_download, snapshot_download

from common import corpus
from common.models import GGUF, MODELS, TOKENIZER_ONLY
from common.paths import MODELS_DIR, NORMEN_PARQUET

SKIP = ["onnx/*", "openvino/*", "coreml/*", "*.onnx", "*.h5", "*.ot", "*.msgpack",
        "tf_model*", "flax_model*", "rust_model*", "*.tflite", "pytorch_model.bin", "*.pth"]
TOKENIZER_FILES = ["tokenizer*", "vocab*", "merges.txt", "special_tokens_map.json", "config.json"]


def main(check: bool) -> None:
    print("Gesetzeskorpus:", "vorhanden" if NORMEN_PARQUET.exists() else "fehlt")
    if not NORMEN_PARQUET.exists() and not check:
        df = corpus.build()
        print(f"  gebaut: {len(df)} Normen")
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    rev_file = MODELS_DIR / "revisions.json"
    revs = json.loads(rev_file.read_text()) if rev_file.exists() else {}
    for name, repo in MODELS.items():
        target = MODELS_DIR / name
        if check:
            print(f"{name:20s}", "vorhanden" if target.exists() else "fehlt")
            continue
        rev = revs.get(name) or HfApi().model_info(repo).sha
        allow = TOKENIZER_FILES if name in TOKENIZER_ONLY else None
        skip = SKIP
        if allow is None and not any(f.endswith(".safetensors") for f in HfApi().list_repo_files(repo, revision=rev)):
            skip = [s for s in SKIP if s != "pytorch_model.bin"]  # z. B. CLIP: nur pytorch_model.bin
        snapshot_download(repo, revision=rev, local_dir=target, allow_patterns=allow,
                          ignore_patterns=None if allow else skip)
        revs[name] = rev
        print(f"{name:20s} {repo}@{rev[:8]}")
    for name, (repo, fname) in GGUF.items():
        target = MODELS_DIR / name
        if check:
            print(f"{name:20s}", "vorhanden" if (target / fname).exists() else "fehlt")
            continue
        rev = revs.get(name) or HfApi().model_info(repo).sha
        hf_hub_download(repo, fname, revision=rev, local_dir=target)
        revs[name] = rev
        print(f"{name:20s} {repo}/{fname}@{rev[:8]}")
    if not check:
        rev_file.write_text(json.dumps(revs, indent=2))


if __name__ == "__main__":
    main("--check" in sys.argv)
