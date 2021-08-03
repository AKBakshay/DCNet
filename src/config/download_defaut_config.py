from pathlib import Path

BASE_DIR = Path("src")

# configs corresponds to be provided files in the drive
config = {
    "model_weight": "1eBY5Gnif_KexV2MMjAvJDSu_SWFIgZoh",
    "dataset": "110yZFcc5AlPzTP57GiQaetKZo0p6OUR8",
}

file_path = {
    "model_weight": BASE_DIR / "pretrain/model/model.pt",
    "dataset": BASE_DIR / "dataset",
}
