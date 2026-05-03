import json, sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

nb = json.load(open("notebooks/16_RFE.ipynb", "r", encoding="utf-8"))
for i, c in enumerate(nb["cells"]):
    src = "".join(c["source"])[:200].replace("\n", " ")
    print(f"[{i:02d}] {c['cell_type']:8s}: {src!r}")
