"""Download selected JASPAR matrices by ID and save them locally."""
import os
import requests

OUT_DIR = os.path.join(os.path.dirname(__file__), "jaspar")
BASE = "https://jaspar.elixir.no/api/v1/matrix/{mid}/?format=jaspar"

# Candidate list: edit after you look at the IC/width table below
MOTIF_IDS = ["MA0001.3", "MA0002.2", "MA0003.4", "MA0004.1", "MA0006.1",
             "MA0007.3", "MA0009.2", "MA0014.3", "MA0017.2", "MA0018.4"]

def download(mid):
    path = os.path.join(OUT_DIR, f"{mid}.jaspar")
    if os.path.exists(path):
        return path
    r = requests.get(BASE.format(mid=mid), timeout=30)
    r.raise_for_status()
    with open(path, "w") as f:
        f.write(r.text)
    return path

if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)
    for mid in MOTIF_IDS:
        try:
            print("saved", download(mid))
        except Exception as e:
            print("FAILED", mid, e)