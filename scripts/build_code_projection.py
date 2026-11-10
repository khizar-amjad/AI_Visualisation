"""Build dashboard/code_projection.csv from real CodeBERT embeddings.

Pipeline: correct_code -> microsoft/codebert-base -> attention-mask-aware mean
pooling -> L2 normalisation -> UMAP (fixed seed) -> problem_id, umap_x, umap_y.

Usage (from the project root, with the optional ML requirements installed):
    python scripts/build_code_projection.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import umap
from transformers import AutoModel, AutoTokenizer

BASE = Path(__file__).resolve().parents[1]
PAIRS_CSV = BASE / "dashboard" / "controlled_code_pairs.csv"
OUT_CSV = BASE / "dashboard" / "code_projection.csv"
PARAMS_JSON = BASE / "dashboard" / "code_projection_params.json"

MODEL_NAME = "microsoft/codebert-base"
MAX_LENGTH = 512
BATCH_SIZE = 8
EXPECTED_PROBLEMS = 178
UMAP_PARAMS = {
    "n_neighbors": 15,
    "min_dist": 0.1,
    "n_components": 2,
    "metric": "cosine",
    "random_state": 42,
}


def embed(codes: list[str]) -> np.ndarray:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME).eval()
    chunks = []
    with torch.no_grad():
        for start in range(0, len(codes), BATCH_SIZE):
            batch = tokenizer(
                codes[start:start + BATCH_SIZE],
                padding=True,
                truncation=True,
                max_length=MAX_LENGTH,
                return_tensors="pt",
            )
            hidden = model(**batch).last_hidden_state
            mask = batch["attention_mask"].unsqueeze(-1).to(hidden.dtype)
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
            chunks.append(torch.nn.functional.normalize(pooled, p=2, dim=1))
            print(f"embedded {min(start + BATCH_SIZE, len(codes))}/{len(codes)}")
    return torch.cat(chunks).numpy()


def main() -> None:
    pairs = pd.read_csv(PAIRS_CSV)
    pairs = pairs[["problem_id", "correct_code"]]
    assert len(pairs) == EXPECTED_PROBLEMS, f"expected {EXPECTED_PROBLEMS}, got {len(pairs)}"
    assert pairs["problem_id"].is_unique, "problem_id values are not unique"
    assert pairs["correct_code"].notna().all(), "missing correct_code"

    embeddings = embed(pairs["correct_code"].astype(str).tolist())
    assert np.isfinite(embeddings).all(), "non-finite embeddings"

    coords = umap.UMAP(**UMAP_PARAMS).fit_transform(embeddings)

    out = pd.DataFrame(
        {"problem_id": pairs["problem_id"].values, "umap_x": coords[:, 0], "umap_y": coords[:, 1]}
    )
    assert len(out) == EXPECTED_PROBLEMS
    assert out["problem_id"].is_unique
    assert np.isfinite(out[["umap_x", "umap_y"]].to_numpy()).all()

    out.to_csv(OUT_CSV, index=False)
    PARAMS_JSON.write_text(
        json.dumps(
            {
                "model": MODEL_NAME,
                "pooling": "attention-mask mean, L2-normalised",
                "max_length": MAX_LENGTH,
                "umap": UMAP_PARAMS,
                "n_problems": len(out),
                "torch": torch.__version__,
                "umap_learn": umap.__version__,
            },
            indent=2,
        )
    )
    print(f"wrote {OUT_CSV} ({len(out)} rows) and {PARAMS_JSON.name}")


if __name__ == "__main__":
    main()
