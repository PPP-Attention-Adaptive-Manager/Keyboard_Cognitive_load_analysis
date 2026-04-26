from pathlib import Path


# project root = folder containing V2
PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "v2" / "models"

def data_path(*parts):
    return DATA_DIR.joinpath(*parts)

def model_path(*parts):
    return MODEL_DIR.joinpath(*parts)