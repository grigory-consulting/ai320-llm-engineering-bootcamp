"""Modelle der Labs (Hugging Face Hub). prefetch.py lädt sie nach labs/models/<ordner>/."""

MODELS = {
    # Lab 2: Vergleichs-Tokenizer (nur Tokenizer-Dateien)
    "gpt2": "openai-community/gpt2",
    "gbert-base": "deepset/gbert-base",
    # Lab 4: Klassifikation Paragraph -> Rechtsgebiet
    "distilbert-german": "distilbert/distilbert-base-german-cased",
    # Lab 6: LoRA vs. Full-Fine-Tuning, Quantisierung
    "smollm2-135m": "HuggingFaceTB/SmolLM2-135M",
    # Lab 5: Retrieval
    "e5-small": "intfloat/multilingual-e5-small",
    "reranker": "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1",
    # Lab 6 (Teil XI): CLIP zero-shot
    "clip-b32": "openai/clip-vit-base-patch32",
}
TOKENIZER_ONLY = {"gpt2", "gbert-base"}

# Lab 7: fertiges GGUF für llama-server auf CPU
GGUF = {"qwen2.5-0.5b-instruct": ("Qwen/Qwen2.5-0.5B-Instruct-GGUF", "qwen2.5-0.5b-instruct-q4_k_m.gguf")}
