
import os, sys
sys.stdout = open(r"C:\Users\MASTER CORE TI\project\poultry_paper\download_log.txt", "w")
sys.stderr = sys.stdout

from datasets import load_dataset
from PIL import Image

save_dir = r"C:\Users\MASTER CORE TI\project\poultry_paper\data\images"
os.makedirs(save_dir, exist_ok=True)

print("Loading dataset from HuggingFace...", flush=True)
ds = load_dataset("Dianyo/poultry-fecal-fl")
print("Download complete", flush=True)

for split_name in ['train', 'test']:
    split = ds[split_name]
    label_names = split.features['label'].names
    print(f"Processing {split_name} ({len(split)} samples)...", flush=True)
    
    for i in range(len(split)):
        sample = split[i]
        label = label_names[sample['label']]
        img = sample['image']
        
        split_dir = os.path.join(save_dir, split_name, label)
        os.makedirs(split_dir, exist_ok=True)
        
        img_path = os.path.join(split_dir, f"{i:05d}.jpg")
        if not os.path.exists(img_path):
            img.save(img_path)
        
        if i % 100 == 0:
            print(f"  {split_name}: {i}/{len(split)}", flush=True)
    
    print(f"  {split_name} done: {len(split)} images", flush=True)

print("ALL DONE", flush=True)
sys.stdout.flush()
