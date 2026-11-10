"""Детерминированный «энкодер» для тестов без загрузки нейросети (хеши основ слов)."""
import re
import zlib

import numpy as np


class HashEncoder:
    dim = 512

    def encode(self, texts, normalize_embeddings=True, **kwargs):
        out = np.zeros((len(texts), self.dim))
        for i, t in enumerate(texts):
            for w in re.findall(r"[а-яёa-z0-9]+", t.lower()):
                out[i, zlib.crc32(w[:5].encode()) % self.dim] += 1.0
        if normalize_embeddings:
            norms = np.linalg.norm(out, axis=1, keepdims=True)
            out = out / np.where(norms == 0, 1, norms)
        return out
