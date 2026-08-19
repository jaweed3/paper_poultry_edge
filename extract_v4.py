
import os, sys, time
from datasets import load_dataset
from PIL import Image

log = open(os.path.join(os.environ["USERPROFILE"], "project", "poultry_paper", "extract_log3.txt"), "w", buffering=1)
sys.stdout = log
sys.stderr = log

print("Starting at " + time.strftime("%H:%M:%S"), flush=True)

save_dir = os.path.join(os.environ["USERPROFILE"], "project", "poultry_paper", "data", "images")
os.makedirs(save_dir, exist_ok=True)

print("Loading dataset...", flush=True)
ds = load_dataset("Dianyo/poultry-fecal-fl")
print("Loaded at " + time.strftime("%H:%M:%S"), flush=True)

# Check label feature
label_feat = ds["train"].features["label"]
print("Label feature type: " + str(type(label_feat)), flush=True)
print("Label feature: " + str(label_feat), flush=True)

# Get labels - try different approaches
if hasattr(label_feat, "names"):
    labels = label_feat.names
elif hasattr(label_feat, "num_classes"):
    labels = [str(i) for i in range(label_feat.num_classes)]
else:
    # Get unique labels from first 100 samples
    labels_set = set()
    for i in range(min(100, len(ds["train"]))):
        labels_set.add(ds["train"][i]["label"])
    labels = sorted(labels_set)
    labels = [str(l) for l in labels]

print("Labels: " + str(labels), flush=True)

# Map integer labels to names
label_map = {}
for i, name in enumerate(labels):
    label_map[i] = name
# Also try string mapping
sample_label = ds["train"][0]["label"]
print("Sample label value: " + str(sample_label) + " type: " + str(type(sample_label)), flush=True)

if isinstance(sample_label, str):
    label_map = {name: name for name in labels}
    label_map_inv = {name: name for name in labels}
elif isinstance(sample_label, int):
    label_map = {i: name for i, name in enumerate(labels)}
else:
    label_map = {i: str(i) for i in range(len(labels))}

print("Label map: " + str(label_map), flush=True)

# Process train split
train_dir = os.path.join(save_dir, "train")
print("Processing train (" + str(len(ds["train"])) + ")...", flush=True)
for i in range(len(ds["train"])):
    s = ds["train"][i]
    lbl_val = s["label"]
    if isinstance(lbl_val, str):
        lbl = lbl_val
    elif isinstance(lbl_val, int):
        lbl = label_map.get(lbl_val, str(lbl_val))
    else:
        lbl = str(lbl_val)
    
    d = os.path.join(train_dir, lbl)
    os.makedirs(d, exist_ok=True)
    p = os.path.join(d, str(i).zfill(5) + ".jpg")
    if not os.path.exists(p):
        img = s["image"]
        if img.mode != "RGB":
            img = img.convert("RGB")
        img.save(p, "JPEG", quality=95)
    if i % 1000 == 0:
        print("  train: " + str(i) + "/" + str(len(ds["train"])), flush=True)

# Process test split
test_dir = os.path.join(save_dir, "test")
print("Processing test (" + str(len(ds["test"])) + ")...", flush=True)
for i in range(len(ds["test"])):
    s = ds["test"][i]
    lbl_val = s["label"]
    if isinstance(lbl_val, str):
        lbl = lbl_val
    elif isinstance(lbl_val, int):
        lbl = label_map.get(lbl_val, str(lbl_val))
    else:
        lbl = str(lbl_val)
    
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

# Summary
print("\nSummary:", flush=True)
for split in ["train", "test"]:
    sp = os.path.join(save_dir, split)
    if os.path.exists(sp):
        for c in os.listdir(sp):
            cp = os.path.join(sp, c)
            if os.path.isdir(cp):
                print("  " + split + "/" + c + ": " + str(len(os.listdir(cp))), flush=True)

print("DONE at " + time.strftime("%H:%M:%S"), flush=True)
log.flush()
log.close()
