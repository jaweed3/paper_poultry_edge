import subprocess, os
subprocess.run(["powershell", "-Command", "Get-Process python -ErrorAction SilentlyContinue | Stop-Process -Force"], capture_output=True)
print("Cleaned up")
base = os.path.join(os.environ["USERPROFILE"], "project", "poultry_paper")
data_dir = os.path.join(base, "data", "ai4d")
os.makedirs(data_dir, exist_ok=True)
r = subprocess.run(["git", "clone", "--depth", "1", "https://github.com/ezinne359/AI4D-Poultry-Dataset.git", data_dir], capture_output=True, text=True, timeout=120)
print("Clone:", "OK" if r.returncode == 0 else r.stderr[:200])
if r.returncode == 0:
    for item in os.listdir(data_dir):
        p = os.path.join(data_dir, item)
        if os.path.isdir(p) and not item.startswith("."):
            print("  " + item + "/ (" + str(len([x for x in os.listdir(p) if not x.startswith(".")])) + " items)")
            for c in os.listdir(p)[:5]:
                cp = os.path.join(p, c)
                if os.path.isdir(cp):
                    print("    " + c + "/ (" + str(len([x for x in os.listdir(cp) if not x.startswith(".")])) + " files)")

