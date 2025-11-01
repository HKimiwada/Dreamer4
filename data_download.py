import os
import shutil
import time
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download
from dotenv import load_dotenv

# Load HF_TOKEN from .env
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    raise RuntimeError("HF_TOKEN not found in environment (.env)")

# Set longer timeout via environment variable
os.environ["HF_HUB_READ_TIMEOUT"] = "300"

# Config
REPO_ID = "zhwang4ai/OpenAI-Minecraft-Contractor"
REPO_TYPE = "dataset"
OUT_DIR = Path("vpt_mvs")
SIZE_LIMIT_GB = 40
SIZE_LIMIT = SIZE_LIMIT_GB * (1024 ** 3)

OUT_DIR.mkdir(parents=True, exist_ok=True)

api = HfApi(token=HF_TOKEN)

def get_free_space_gb(path="."):
    """Return free disk space (in GB) at given path."""
    total, used, free = shutil.disk_usage(path)
    return free / (1024**3)

def download_one_file(filename: str, dest_path: Path):
    """Download a single file from the HF hub and move it to dest_path."""
    local_path = hf_hub_download(
        repo_id=REPO_ID,
        filename=filename,
        repo_type=REPO_TYPE,
        token=HF_TOKEN
    )
    shutil.copy(local_path, dest_path)
    print(f"[DOWNLOADED] {filename} → {dest_path}")

def list_repo_files_with_retry(api, repo_id, repo_type, max_retries=5):
    """List repo files with retry logic."""
    for attempt in range(max_retries):
        try:
            return api.list_repo_files(repo_id=repo_id, repo_type=repo_type)
        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = 5 * (2 ** attempt)
                print(f"[WARN] Attempt {attempt+1} failed: {e}. Retrying in {wait_time}s...")
                time.sleep(wait_time)
            else:
                raise

def main():
    # 1) List files in the repo (video & action logs)
    print("[INFO] Listing files in dataset repo …")
    files = list_repo_files_with_retry(api, REPO_ID, REPO_TYPE)
    videos = sorted([f for f in files if f.endswith(".mp4")])
    actions = sorted([f for f in files if f.endswith(".jsonl")])

    print(f"[INFO] Found {len(videos)} videos, {len(actions)} action-logs")
    num = min(len(videos), len(actions))
    print(f"[INFO] Will download up to {num} recordings (≈ {SIZE_LIMIT_GB} GB limit)")

    total_bytes = 0
    for i in range(num):
        vid = videos[i]
        act = actions[i]
        free_gb = get_free_space_gb(OUT_DIR)
        if free_gb < 10:  # stop early if less than 10 GB free
            print(f"[WARN] Only {free_gb:.2f} GB free — stopping to avoid filling disk.")
            break
        print(f"[INFO] Downloading pair {i+1}/{num}: {vid} + {act}")

        vid_dest = OUT_DIR / Path(vid).name
        act_dest = OUT_DIR / Path(act).name

        download_one_file(vid, vid_dest)
        download_one_file(act, act_dest)

        size_vid = vid_dest.stat().st_size
        size_act = act_dest.stat().st_size
        total_bytes += (size_vid + size_act)

        print(f"[INFO] Downloaded {i+1} pairs — total size ≈ {total_bytes/(1024**3):.2f} GB")

        if total_bytes >= SIZE_LIMIT:
            print(f"[INFO] Reached size limit ~{SIZE_LIMIT_GB} GB — stopping early.")
            break

    print(f"[DONE] Downloaded {i+1} recordings, total size is approx {total_bytes/(1024**3):.2f} GB in folder {OUT_DIR}")

if __name__ == "__main__":
    main()