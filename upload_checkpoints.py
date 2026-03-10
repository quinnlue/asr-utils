"""Upload selected checkpoints + final model to HF Hub."""

from huggingface_hub import HfApi

REPO_ID = "lueqin/whisper-medium-finetune"  # ← change if needed
BASE_DIR = "whisper-medium-finetune"
FINAL_DIR = "whisper-medium-finetune-final"
CHECKPOINTS = [10000, 12500, 13500, 15500, 18000]

api = HfApi()
api.create_repo(REPO_ID, exist_ok=True)

# upload final model
print(f"Uploading {FINAL_DIR} → {REPO_ID} (main branch)")
api.upload_folder(folder_path=FINAL_DIR, repo_id=REPO_ID)

# upload selected checkpoints as subfolders
for step in CHECKPOINTS:
    ckpt = f"checkpoint-{step}"
    local = f"{BASE_DIR}/{ckpt}"
    print(f"Uploading {local} → {REPO_ID}/{ckpt}")
    api.upload_folder(folder_path=local, repo_id=REPO_ID, path_in_repo=ckpt)

print("All done!")
