from pathlib import Path


# project root = folder containing V2
PROJECT_ROOT = Path(__file__).resolve().parents[2]

V2_DIR = PROJECT_ROOT / "V2"

DATA_DIR = V2_DIR / "data"
MODEL_DIR = V2_DIR / "models"

def data_path(*parts):
    return DATA_DIR.joinpath(*parts)

def model_path(*parts):
    return MODEL_DIR.joinpath(*parts)