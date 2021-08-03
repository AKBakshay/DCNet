from pathlib import Path

BASE_DIR = Path("src")

# configs corresponds to be provided files in the drive
config = {
    "model_weight": "1FJsxrulia9KYstv5zRcmnlv1NJX6ocbF",
    "dataset": "1W9A2GoUTxe0-QzASnV-LlpZhfY8E9tZn",
}

file_path = {
    "model_weight": BASE_DIR / "pretrain/model/model.pt",
    "dataset": BASE_DIR / "dataset.tar.gz",
}
