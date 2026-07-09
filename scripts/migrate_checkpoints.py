import os
import re

OLD = re.compile(r"^(\w+)_model_(\d{4})-(\d{2})-(\d{2})_(\d{2})_(\d{2})_(\d{2})$")

for stem in sorted(os.listdir("checkpoint")):
    m = OLD.match(stem)
    if not m:
        continue
    name, Y, Mo, D, h, mi, s = m.groups()
    new_stem = f"{Y}{Mo}{D}_{h}{mi}_{name}"
    old_path = os.path.join("checkpoint", stem)
    new_path = os.path.join("checkpoint", new_stem)
    os.rename(old_path, new_path)
    print(f"  {stem}  ->  {new_stem}")

print("Done.")
