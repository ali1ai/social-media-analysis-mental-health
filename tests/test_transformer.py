"""Exercises the transformer training/inference loop with a tiny, randomly
initialised model and a locally-trained tokenizer (no downloads)."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
tokenizers = pytest.importorskip("tokenizers")

from socialmediamind.transformer import finetune


def _tiny_model_and_tokenizer(texts, n_labels):
    from tokenizers import Tokenizer, models, pre_tokenizers, trainers
    from transformers import BertConfig, BertForSequenceClassification, PreTrainedTokenizerFast

    tok = Tokenizer(models.WordLevel(unk_token="[UNK]"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    tok.train_from_iterator(texts, trainers.WordLevelTrainer(special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]"]))
    hf_tok = PreTrainedTokenizerFast(tokenizer_object=tok, pad_token="[PAD]", unk_token="[UNK]")
    cfg = BertConfig(
        vocab_size=hf_tok.vocab_size, hidden_size=32, num_hidden_layers=1, num_attention_heads=2,
        intermediate_size=64, max_position_embeddings=64, num_labels=n_labels,
    )
    return BertForSequenceClassification(cfg), hf_tok


def test_finetune_learns_separable_toy_task():
    rng = np.random.default_rng(0)
    labels = np.array(["calm", "worried"])
    vocab = {"calm": "sunny walk coffee friends", "worried": "panic racing fear nervous"}
    y = rng.integers(0, 2, 200)
    texts = [" ".join(rng.choice(vocab[labels[k]].split(), 6)) for k in y]
    model, tok = _tiny_model_and_tokenizer(texts, 2)
    clf = finetune(texts, labels[y], labels, model=model, tokenizer=tok, epochs=8, batch_size=16,
                   lr=3e-3, max_length=16, device="cpu", log_every=0)
    proba = clf.predict_proba(texts)
    assert proba.shape == (200, 2)
    np.testing.assert_allclose(proba.sum(1), 1, rtol=1e-5)
    assert (clf.predict(texts) == labels[y]).mean() > 0.9
