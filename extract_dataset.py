
import os, sys
from datasets import load_dataset
from PIL import Image

sys.stdout = open(os.path.join(os.environ["USERPROFILE"], "project", "poultry_paper", "extract_log.txt"), "w", buffering=1)
sys.stderr = sys.stdout

save_dir = os.path.join(os.environ["USERPROFILE"], "project", "poultry_paper", "data", "images")
os.makedirs(save_dir, exist_ok=True)

print("Loading dataset...", flush=True)
ds = load_dataset("Dianyo/poultry-fecal-fl")
print("Dataset loaded", flush=True)

label_names = ds["train"].features["label"].names
print("Labels: " + str(label_names), flush=True)

for split_name in ["train", "test"]:
    split = ds[split_name]
    print(f"Processing {split_name} ({len(split)} samples)...", flush=True)
    
    for i in range(len(split)):
        sample = split[i]
        label = label_names[sample["label"]]
        img = sample["image"]
        
        split_dir = os.path.join(save_dir, split_name, label)
        os.makedirs(split_dir, exist_ok=True)
        
        img_path = os.path.join(split_dir, f"{i:05d}.jpg")
        if not os.path.exists(img_path):
            if img.mode != "RGB":
                img = img.convert("RGB")
            img.save(img_path, "JPEG", quality=95)
        
        if i % 200 == 0:
            print(f"  {split_name}: {i}/{len(split)}", flush=True)
    
    count = sum(len(os.listdir(os.path.join(save_dir, split_name, c))) for c in os.listdir(os.path.join(save_dir, split_name)) if os.path.isdir(os.path.join(save_dir, split_name, c)))
    print(f"  {split_name} done: {count} images", flush=True)

print("ALL DONE", flush=True)
