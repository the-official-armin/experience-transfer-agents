"""Embedding-based retrieval for the `retrieved` condition.

Uses a small, separate, open-weight embedding model
(sentence-transformers/all-MiniLM-L6-v2, ~22M params, Apache-2.0) for the
experience-bank similarity search -- deliberately NOT GLM-4.7-Flash.
CLAUDE.md's single-model policy applies to generation (agent +
representations), not retrieval; embedding and generation are different
roles. Runs entirely locally (no API call), which also sidesteps the Z.ai
endpoint instability seen elsewhere in this project for this step
specifically.

Retrieval method (MiniLM cosine similarity over the `raw` representation) is
fixed here BEFORE any evaluation, per CLAUDE.md's hard constraint -- do not
tune this after seeing results.
"""

import numpy as np
from sentence_transformers import SentenceTransformer

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

_model = None


def get_embedding_model():
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _model


def embed_texts(texts):
    return get_embedding_model().encode(texts, normalize_embeddings=True)


def retrieve_best_experience(task, experiences, representation="raw"):
    """Fixed retrieval method: cosine similarity between the task's goal and
    each experience's representation text. Returns (best_experience, score)."""
    model = get_embedding_model()
    task_embedding = model.encode([task["goal"]], normalize_embeddings=True)[0]
    experience_texts = [e["representations"][representation] for e in experiences]
    experience_embeddings = model.encode(experience_texts, normalize_embeddings=True)

    similarities = experience_embeddings @ task_embedding
    best_index = int(np.argmax(similarities))
    return experiences[best_index], float(similarities[best_index])
