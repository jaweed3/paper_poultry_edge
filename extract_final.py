import os, sys, time
from datasets import load_dataset
from PIL import Image

log = open(os.path.join(os.environ["USERPROFILE"], "project", "poultry_paper", "extract_final.txt"), "w", buffering=1)
sys.stdout = log
sys.stderr = log

print("Starting at " + time.strftime("%H:%M:%S"), flush=True)

save_dir = os.path.join(os.environ["USERPROFILE"], "project", "poultry_paper", "data", "images")
os.makedirs(save_dir, exist_ok=True)

print("Loading dataset...", flush=True)
ds = load_dataset("Dianyo/poultry-fecal-fl")
print("Loaded! Train=" + str(len(ds["train"])) + " Test=" + str(len(ds["test"])), flush=True)

# Label mapping: int -> class name (from Zenodo description)
LABEL_MAP = {0: "cocci", 1: "healthy", 2: "ncd", 3: "salmo"}
print("Label map: " + str(LABEL_MAP), flush=True)

for split_name in ["train", "test"]:
    split = ds[split_name]
    out_dir = os.path.join(save_dir, split_name)
    print("Processing " + split_name + " (" + str(len(split)) + ")...", flush=True)
    
    for i in range(len(split)):
        s = split[i]
        lbl = LABEL_MAP.get(s["label"], "unknown")
        cls_dir = os.path.join(out_dir, lbl)
        os.makedirs(cls_dir, exist_ok=True)
        
        img_path = os.path.join(cls_dir, str(i).zfill(5) + ".jpg")
        if not os.path.exists(img_path):
            img = s["image"]
            if img.mode != "RGB":
                img = img.convert("RGB")
            img.save(img_path, "JPEG", quality=95)
        
        if i % 1000 == 0:
            print("  " + split_name + ": " + str(i) + "/" + str(len(split)), flush=True)
    
    # Count
    total = 0
    for c in os.listdir(out_dir):
        cp = os.path.join(out_dir, c)
        if os.path.isdir(cp):
            n = len(os.listdir(cp))
            total += n
            print("  " + split_name + "/" + c + ": " + str(n), flush=True)
    print("  " + split_name + " TOTAL: " + str(total), flush=True)

print("DONE at " + time.strftime("%H:%M:%S"), flush=True)
log.flush()
log.close()
