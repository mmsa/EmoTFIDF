"""Whole-sentence emotion scoring with the bundled GoEmotions transformer."""

import json
import re
from pathlib import Path

import numpy as np
import torch
from torch import nn


_LABELS = ("joy", "sadness", "anger", "fear", "disgust", "surprise", "neutral")
_ASSETS = Path(__file__).parent / "sentence_assets"
_TOKEN = re.compile(r"[a-z]+(?:'[a-z]+)?|[0-9]+|[^\w\s]")


class _Encoder(nn.Module):
    def __init__(self, vocabulary_size):
        super().__init__()
        self.embed = nn.Embedding(vocabulary_size, 48, padding_idx=0)
        self.pos = nn.Parameter(torch.empty(48, 48))
        layer = nn.TransformerEncoderLayer(48, 4, 96, 0.15, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, 1)
        self.out = nn.Linear(48, len(_LABELS))

    def forward(self, tokens):
        hidden = self.embed(tokens) + self.pos[:tokens.shape[1]]
        return self.out(self.encoder(hidden)[:, 0])


class SentenceEmotionTransformer:
    """Load once and call score(text) for seven independent emotion scores."""

    def __init__(self):
        self._vocab = json.loads((_ASSETS / "vocab.json").read_text(encoding="utf-8"))
        self._model = _Encoder(len(self._vocab) + 3)
        with np.load(_ASSETS / "weights.npz", allow_pickle=False) as weights:
            self._model.load_state_dict({k: torch.from_numpy(weights[k]) for k in weights})
        self._model.eval()

    @torch.no_grad()
    def score(self, text):
        """Score the first 47 tokens of an English string using the bundled weights."""
        if not isinstance(text, str):
            raise TypeError("text must be a string")
        words = _TOKEN.findall(text.lower().replace("’", "'"))
        ids = [2] + [self._vocab.get(word, 1) for word in words[:47]]
        values = self._model(torch.tensor([ids])).sigmoid()[0].tolist()
        return dict(zip(_LABELS, values))
