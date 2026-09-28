import os
import shutil
from pathlib import Path

workdir = Path(r"C:\Users\erikh\OneDrive\Документы\Lyra0.1")
backup_dir = workdir / "backup_248m"

if backup_dir.exists():
    shutil.rmtree(backup_dir)

backup_dir.mkdir(parents=True, exist_ok=True)

# Copy directory trees
for dir_name in ["lyra", "configs", "data", "artifacts"]:
    src = workdir / dir_name
    dst = backup_dir / dir_name
    if src.exists():
        shutil.copytree(src, dst, ignore=shutil.ignore_patterns("*.pyc", "__pycache__"))
        print(f"Copied {dir_name}")

# Copy specific files  
for file_path in [workdir / "train.py", workdir / "scripts" / "train.py", workdir / "scripts" / "model_info.py"]:
    if file_path.exists():
        shutil.copy2(file_path, backup_dir / file_path.name)
        print(f"Copied {file_path.name}")

# Copy config files
for file_name in ["lyra_0_1.json", "kaggle.json", "colab.json", "debug.json"]:
    src = workdir / file_name
    if src.exists():
        shutil.copy2(src, backup_dir / file_name)
        print(f"Copied {file_name}")

print(f"\nBackup created at: {backup_dir}")
print(f"Backup contents count: {len(list(backup_dir.rglob('*')))}")