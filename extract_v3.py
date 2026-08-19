
import os, sys, time
from datasets import load_dataset
from PIL import Image

log = open(os.path.join(os.environ["USERPROFILE"], "project", "poultry_paper", "extract_log2.txt"), "w", buffering=1)
sys.stdout = log
sys.stderr = log

print("Starting at " + time.strftime("%H:%M:%S"), flush=True)

save_dir = os.path.join(os.environ["USERPROFILE"], "project", "poultry_paper", "data", "images")
os.makedirs(save_dir, exist_ok=True)

print("Loading dataset...", flush=True)
ds = load_dataset("Dianyo/poultry-fecal-fl", cache_dir=os.path.join(os.environ["USERPROFILE"], ".cache", "huggingface"))
print("Loaded at " + time.strftime("%H:%M:%S"), flush=True)

labels = ds["train"].features["label"].names
print("Labels: " + str(labels), flush=True)

# Process train split
train_dir = os.path.join(save_dir, "train")
print("Processing train (" + str(len(ds["train"])) + ")...", flush=True)
for i in range(len(ds["train"])):
    s = ds["train"][i]
    lbl = labels[s["label"]]
    d = os.path.join(train_dir, lbl)
    os.makedirs(d, exist_ok=True)
    p = os.path.join(d, str(i).zfill(5) + ".jpg")
    if not os.path.exists(p):
        img = s["image"]
        if img.mode != "RGB":
            img = img.convert("RGB")
        img.save(p, "JPEG", quality=95)
    if i % 500 == 0:
        print("  train: " + str(i) + "/" + str(len(ds["train"])), flush=True)

# Process test split
test_dir = os.path.join(save_dir, "test")
print("Processing test (" + str(len(ds["test"])) + ")...", flush=True)
for i in range(len(ds["test"])):
    s = ds["test"][i]
    lbl = labels[s["label"]]
    d = os.path.join(test_dir, lbl)
    os.makedirs(d, exist_ok=True)
    p = os.path.join(d, str(i).zfill(5) + ".jpg")
    if not os.path.exists(p):
        img = s["image"]
        if img.mode != "RGB":
            img = img.convert("RGB")
        img.save(p, "JPEG", quality=95)
    if i % 500 == 0:
        print("  test: " + str(i) + "/" + str(len(ds["test"])), flush=True)

print("DONE at " + time.strftime("%H:%M:%S"), flush=True)
log.flush()
log.close()
