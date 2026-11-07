import numpy as np
from jaspar_loader import information_content

STRENGTH_FRACTIONS = [0.2, 0.4, 0.6, 0.7, 0.8, 0.9, 1.0]  # fraction of original total IC

def blend(ppm, alpha):
    return (1 - alpha) * ppm + alpha * 0.25

def ppm_at_fraction(ppm, fraction):
    """Find alpha so blended IC = fraction * original IC (bisection)."""
    target = fraction * information_content(ppm)
    lo, hi = 0.0, 1.0
    for _ in range(60):
        mid = (lo + hi) / 2
        if information_content(blend(ppm, mid)) > target:
            lo = mid   # still too strong, blend more
        else:
            hi = mid
    alpha = (lo + hi) / 2
    out = blend(ppm, alpha)
    return out, alpha, information_content(out)

if __name__ == "__main__":
    import pandas as pd
    from jaspar_loader import load_motif, get_ppm, JASPAR_DIR
    from selected_motifs import SELECTED
    import os
    rows = []
    for mid in SELECTED:
        ppm = get_ppm(load_motif(os.path.join(JASPAR_DIR, f"{mid}.jaspar")))
        for f in STRENGTH_FRACTIONS:
            _, a, ic = ppm_at_fraction(ppm, f)
            rows.append(dict(motif=mid, fraction=f, alpha=round(a, 3),
                             total_ic=round(ic, 2),
                             ic_per_pos=round(ic / ppm.shape[1], 2)))
    print(pd.DataFrame(rows).to_string(index=False))