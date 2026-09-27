from . import modules
from pathlib import Path
from . import scheduler
from .interface import Interface
from .modules.transformer import VampNet


__version__ = "0.0.1"

ROOT = Path(__file__).parent.parent
MODELS_DIR = ROOT / "models" / "unloop"

# Remove Hugging Face download logic
# from huggingface_hub import hf_hub_download, HfFileSystem
# DEFAULT_HF_MODEL_REPO_DIR = ROOT / "DEFAULT_HF_MODEL_REPO"
# DEFAULT_HF_MODEL_REPO = DEFAULT_HF_MODEL_REPO_DIR.read_text().strip()
# FS = HfFileSystem()

def download_codec():
    # from dac.model.dac import DAC
    from lac.model.lac import LAC as DAC
    # In this version, we don't download from Hugging Face
    codec_path = MODELS_DIR / "codec.pth"
    if not codec_path.exists():
        print(f"{codec_path} does not exist, please manually download it.")
        # You can replace this with logic to load a local model or alert the user to download manually.
        raise FileNotFoundError(f"{codec_path} is missing. Please download it.")
    return codec_path
    

def download_default():
    # Instead of downloading, check if the model files are locally available
    filenames = ["coarse.pth", "c2f.pth", "wavebeat.pth"]
    paths = []
    for filename in filenames:
        path = MODELS_DIR / filename
        if not path.exists():
            print(f"{path} does not exist, please manually download it.")
            # You can replace this with logic to load a local model or alert the user to download manually.
            raise FileNotFoundError(f"{path} is missing. Please download it.")
        paths.append(path)
    
    # load the models
    return paths[0], paths[1]


def download_finetuned(name, repo_id=None):
    # Same as `download_default`, we will check for local files instead of downloading.
    filenames = ["coarse.pth", "c2f.pth"]
    paths = []
    for filename in filenames:
        path = MODELS_DIR / "loras" / name / filename
        if not path.exists():
            print(f"{path} does not exist, please manually download it.")
            raise FileNotFoundError(f"{path} is missing. Please download it.")
        paths.append(path)
    
    # load the models
    return paths[0], paths[1]
    
def list_finetuned(repo_id=None):
    # Instead of listing models on Hugging Face, we check the local directory
    finetuned_dir = MODELS_DIR / "loras"
    if not finetuned_dir.exists():
        print("No fine-tuned models found. Please ensure they are available locally.")
        return []

    valid_diritems = []
    for item in finetuned_dir.iterdir():
        # Check that there are "c2f.pth" and "coarse.pth" files
        if (item / "c2f.pth").exists() and (item / "coarse.pth").exists():
            valid_diritems.append(item.name)

    return valid_diritems
