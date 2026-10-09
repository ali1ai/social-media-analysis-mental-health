"""The notebook embeds a copy of src/socialmediamind so it can run standalone
(Colab/Kaggle, no git clone). This test fails if that copy drifts from src/.
Fix a failure with:  python scripts/build_notebook.py
"""

import base64
import hashlib
import io
import json
import re
import tarfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
NOTEBOOK = REPO / "notebooks" / "SocialMediaMind.ipynb"


def _setup_cell_source() -> str:
    nb = json.loads(NOTEBOOK.read_text())
    for cell in nb["cells"]:
        src = "".join(cell["source"])
        if cell["cell_type"] == "code" and "EMBEDDED_PACKAGE =" in src:
            return src
    raise AssertionError("setup cell with EMBEDDED_PACKAGE not found")


def test_embedded_package_matches_src():
    src = _setup_cell_source()
    blob_b64 = re.search(r'EMBEDDED_PACKAGE = "([A-Za-z0-9+/=]+)"', src).group(1)
    sha = re.search(r'EMBEDDED_SHA256 = "([0-9a-f]{64})"', src).group(1)
    blob = base64.b64decode(blob_b64)
    assert hashlib.sha256(blob).hexdigest() == sha, "checksum in notebook does not match its blob"

    with tarfile.open(fileobj=io.BytesIO(blob), mode="r:gz") as tar:
        embedded = {Path(m.name).name: tar.extractfile(m).read() for m in tar.getmembers() if m.isfile()}
    on_disk = {f.name: f.read_bytes() for f in (REPO / "src" / "socialmediamind").glob("*.py")}
    assert embedded.keys() == on_disk.keys(), "file list differs: run python scripts/build_notebook.py"
    stale = [name for name in on_disk if embedded[name] != on_disk[name]]
    assert not stale, f"embedded copy is stale for {stale}: run python scripts/build_notebook.py"


def test_notebook_never_asks_for_credentials():
    text = NOTEBOOK.read_text()
    for forbidden in ("git clone", "getpass", "<your-username>", "kaggle.json"):
        assert forbidden not in text, f"notebook should not contain {forbidden!r}"


def test_notebook_is_committed_without_outputs():
    nb = json.loads(NOTEBOOK.read_text())
    code = [c for c in nb["cells"] if c["cell_type"] == "code"]
    assert all(not c.get("outputs") for c in code), "clear outputs before committing"
