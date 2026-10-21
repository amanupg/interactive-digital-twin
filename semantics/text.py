"""Text side of the language field: CLIP text encoder + LERF-style relevancy."""

import numpy as np
import open_clip
import torch

from extract_features import CLIP_MODEL

NEGATIVES = ["object", "things", "stuff", "texture"]
_model = _tokenizer = None


def encode_text(prompts):
    global _model, _tokenizer
    if _model is None:
        _model, _, _ = open_clip.create_model_and_transforms(*CLIP_MODEL, device="cpu")
        _model.eval()
        _tokenizer = open_clip.get_tokenizer(CLIP_MODEL[0])
    with torch.no_grad():
        t = _model.encode_text(_tokenizer(prompts))
    return torch.nn.functional.normalize(t, dim=-1).numpy()


def relevancy(feats, query, negatives=NEGATIVES, temp=10.0):
    """LERF relevancy: min over negatives of softmax(query vs negative).

    feats: (N, D) L2-normalised image-side embeddings (rows of zeros allowed).
    Returns (N,) scores in [0, 1]; 0.5 means "no better than a generic word".
    """
    t = encode_text([query] + list(negatives))
    sims = feats.astype(np.float32) @ t.T * temp       # (N, 1 + K)
    q, neg = sims[:, :1], sims[:, 1:]
    pair = np.exp(q) / (np.exp(q) + np.exp(neg))      # (N, K)
    return pair.min(axis=1)
