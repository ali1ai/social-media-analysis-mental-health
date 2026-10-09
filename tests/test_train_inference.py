"""End-to-end: train script -> saved bundle -> load -> predict -> CLI."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from make_synthetic_fixture import make

from socialmediamind.inference import SocialMediaMindModel, main
from socialmediamind.train import run


def test_train_save_load_predict_cli(tmp_path, capsys):
    csv = tmp_path / "fixture.csv"
    make(1500, seed=7).to_csv(csv, index=False)
    metrics = run(str(csv), out_dir=str(tmp_path / "models"), report_dir=str(tmp_path / "reports"), fast=True)

    assert metrics["test"]["macro_f1"] > 0.6
    assert 0.8 <= metrics["conformal"]["coverage"] <= 1.0
    assert (tmp_path / "reports" / "train_metrics.json").exists()

    bundle = tmp_path / "models" / "socialmediamind_bundle.joblib"
    model = SocialMediaMindModel.load(bundle)
    preds = model.predict(["coffee with friends this weekend", "panic heart racing worried"])
    assert {"top_label", "confidence", "prediction_set", "defer_to_human"} <= preds[0].keys()
    assert all(isinstance(p["top_label"], str) for p in preds)

    capsys.readouterr()
    assert main(["--model", str(bundle), "panic and racing heart"]) == 0
    line = capsys.readouterr().out.strip().splitlines()[-1]
    assert json.loads(line)["top_label"] in model.labels
