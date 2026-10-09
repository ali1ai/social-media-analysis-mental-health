"""Optional fine-tuned transformer baseline (requires ``pip install -e .[transformer]``).

Deliberately minimal: a plain PyTorch training loop (no Trainer magic) with
class-weighted cross-entropy, AdamW, linear warm-up/decay and mixed precision
on GPU. The wrapper exposes ``decision_function`` (logits) / ``predict_proba``
so it plugs into exactly the same temperature-scaling and conformal layer as
the linear model, which makes the comparison apples-to-apples.
"""

from __future__ import annotations

import math
import random

import numpy as np


def _require_torch():
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
    except ImportError as e:  # pragma: no cover
        raise ImportError("Install the optional extra: pip install -e '.[transformer]'") from e


class TransformerClassifier:
    def __init__(self, model, tokenizer, labels, max_length: int = 256, device: str | None = None):
        import torch

        self.model = model
        self.tokenizer = tokenizer
        self.classes_ = np.asarray(labels)
        self.max_length = max_length
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def _batches(self, texts, batch_size):
        texts = [str(t) for t in texts]
        for i in range(0, len(texts), batch_size):
            yield texts[i : i + batch_size]

    def decision_function(self, texts, batch_size: int = 64) -> np.ndarray:
        import torch

        self.model.eval()
        out = []
        with torch.no_grad():
            for chunk in self._batches(list(texts), batch_size):
                enc = self.tokenizer(
                    chunk, truncation=True, max_length=self.max_length, padding=True, return_tensors="pt"
                ).to(self.device)
                with torch.autocast(device_type="cuda", enabled=self.device == "cuda"):
                    logits = self.model(**enc).logits
                out.append(logits.float().cpu().numpy())
        return np.concatenate(out) if out else np.empty((0, len(self.classes_)))

    def predict_proba(self, texts) -> np.ndarray:
        z = self.decision_function(texts)
        z = z - z.max(axis=1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(axis=1, keepdims=True)

    def predict(self, texts) -> np.ndarray:
        return self.classes_[self.decision_function(texts).argmax(axis=1)]


def finetune(
    train_texts,
    train_labels,
    labels,
    model_name: str = "distilroberta-base",
    model=None,
    tokenizer=None,
    epochs: int = 2,
    batch_size: int = 32,
    lr: float = 3e-5,
    weight_decay: float = 0.01,
    warmup_ratio: float = 0.06,
    max_length: int = 256,
    class_weighted: bool = True,
    seed: int = 42,
    device: str | None = None,
    log_every: int = 100,
) -> TransformerClassifier:
    """Fine-tune a sequence classifier. Pass ``model``/``tokenizer`` to skip downloading."""
    _require_torch()
    import torch
    from torch.nn import functional as F
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        get_linear_schedule_with_warmup,
    )

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    labels = np.asarray(labels)
    lab2id = {c: i for i, c in enumerate(labels)}
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    if model is None:
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=len(labels),
            id2label={i: str(c) for i, c in enumerate(labels)},
            label2id={str(c): i for i, c in enumerate(labels)},
        )
    clf = TransformerClassifier(model, tokenizer, labels, max_length=max_length, device=device)

    texts = [str(t) for t in train_texts]
    y = torch.tensor([lab2id[v] for v in np.asarray(train_labels)])
    weight = None
    if class_weighted:
        counts = torch.bincount(y, minlength=len(labels)).float().clamp(min=1)
        weight = (counts.sum() / (len(labels) * counts)).to(clf.device)

    steps_per_epoch = math.ceil(len(texts) / batch_size)
    total = steps_per_epoch * epochs
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = get_linear_schedule_with_warmup(opt, int(warmup_ratio * total), total)
    scaler = torch.amp.GradScaler("cuda", enabled=clf.device == "cuda")

    step = 0
    for epoch in range(epochs):
        model.train()
        perm = np.random.permutation(len(texts))
        running = 0.0
        for b in range(steps_per_epoch):
            idx = perm[b * batch_size : (b + 1) * batch_size]
            enc = tokenizer(
                [texts[i] for i in idx], truncation=True, max_length=max_length, padding=True, return_tensors="pt"
            ).to(clf.device)
            with torch.autocast(device_type="cuda", enabled=clf.device == "cuda"):
                logits = model(**enc).logits
            loss = F.cross_entropy(logits.float(), y[idx].to(clf.device), weight=weight)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            running += loss.item()
            step += 1
            if log_every and step % log_every == 0:
                print(f"epoch {epoch + 1}/{epochs} step {step}/{total} loss {running / (b + 1):.4f}")
    return clf
