import os, sys
from datasets import load_dataset

log_path = os.path.join(os.environ["USERPROFILE"], "project", "poultry_paper", "extract_log.txt")
sys.stdout = open(log_path, "w", buffering=1)
sys.stderr = sys.stdout

save_dir = os.path.join(os.environ["USERPROFILE"], "project", "poultry_paper", "data", "images")
os.makedirs(save_dir, exist_ok=True)

print("Loading dataset from cache...", flush=True)
ds = load_dataset("Dianyo/poultry-fecal-fl")
print("Loaded!", flush=True)

label_names = ds["train"].features["label"].names
print("Labels: " + str(label_names), flush=True)

for split_name in ["train", "test"]:
    split = ds[split_name]
    out_dir = os.path.join(save_dir, split_name)
    print("Processing " + split_name + " (" + str(len(split)) + " samples)...", flush=True)
    
    for i in range(len(split)):
        sample = split[i]
        label = label_names[sample["label"]]
        img = sample["image"]
        
        cls_dir = os.path.join(out_dir, label)
        os.makedirs(cls_dir, exist_ok=True)
        
        img_path = os.path.join(cls_dir, str(i).zfill(5) + ".jpg")
        if not os.path.exists(img_path):
            if img.mode != "RGB":
                img = img.convert("RGB")
            img.save(img_path, "JPEG", quality=95)
        
        if i % 200 == 0:
            print("  " + split_name + ": " + str(i) + "/" + str(len(split)), flush=True)
    
    total = sum(len(os.listdir(os.path.join(out_dir, c))) for c in os.listdir(out_dir) if os.path.isdir(os.path.join(out_dir, c)))
    print("  " + split_name + " done: " + str(total) + " images", flush=True)

print("ALL DONE", flush=True)
