"""Build the README graphics in docs/assets/.

Usage (from the repo root):
    python docs/make_readme_art.py

    hero.svg            animated DNA helix banner (CSS animation)
    scan.svg            animated sliding window hunting for a motif
    logo_strength.gif   REAL data: MA0001.3's sequence logo as its strength
                        goes 20% -> 100% (made with data/strength.py)
    algorithms.svg      animated cards for the three search strategies
"""
import io, os, sys
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "docs", "assets")
sys.path.insert(0, os.path.join(ROOT, "data"))      # data/ uses flat imports

BASE_COL = {"A": "#1baf7a", "C": "#2a78d6", "G": "#eda100", "T": "#e34948"}
BG1, BG2, INK, INK2 = "#0f1b2d", "#16263f", "#f0efec", "#9fb3c8"


def _write(name, text):
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, name), "w", encoding="utf-8") as f:
        f.write(text)
    print("wrote", os.path.join("docs", "assets", name))


# ---------------------------------------------------------------- hero.svg
def hero():
    W, H, n = 900, 230, 26
    rng = np.random.default_rng(3)
    letters = rng.choice(list("ACGT"), n)
    comp = {"A": "T", "C": "G", "G": "C", "T": "A"}
    rungs = []
    for i, b in enumerate(letters):
        x = 30 + i * (W - 60) / (n - 1)
        d = f"-{i * 0.16:.2f}s"
        rungs.append(f"""
  <g transform="translate({x:.1f},0)">
    <line class="rung" x1="0" y1="45" x2="0" y2="185" style="animation-delay:{d}"/>
    <g class="top" style="animation-delay:{d}"><circle cx="0" cy="45" r="11" fill="{BASE_COL[b]}"/>
      <text x="0" y="49.5" class="nt">{b}</text></g>
    <g class="bot" style="animation-delay:{d}"><circle cx="0" cy="185" r="11" fill="{BASE_COL[comp[b]]}"/>
      <text x="0" y="189.5" class="nt">{comp[b]}</text></g>
  </g>""")
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">
<defs>
  <linearGradient id="bg" x1="0" y1="0" x2="1" y2="1">
    <stop offset="0" stop-color="{BG1}"/><stop offset="1" stop-color="{BG2}"/></linearGradient>
  <linearGradient id="title" x1="0" x2="1">
    <stop offset="0" stop-color="#5fa8ff"/><stop offset="0.5" stop-color="#3fe0a8"/>
    <stop offset="1" stop-color="#ffb347"/></linearGradient>
</defs>
<style>
  .top {{ animation: up 2.4s ease-in-out infinite; }}
  .bot {{ animation: down 2.4s ease-in-out infinite; }}
  .rung {{ stroke: #ffffff; stroke-opacity: .18; stroke-width: 2;
          transform-box: fill-box; transform-origin: center;
          animation: squash 2.4s ease-in-out infinite; }}
  @keyframes up   {{ 0%,100% {{ transform: translateY(0) }}   50% {{ transform: translateY(140px) }} }}
  @keyframes down {{ 0%,100% {{ transform: translateY(0) }}   50% {{ transform: translateY(-140px) }} }}
  @keyframes squash {{ 0%,100% {{ transform: scaleY(1) }} 50% {{ transform: scaleY(-1) }} }}
  .nt {{ font: 700 12px ui-monospace, Menlo, Consolas, monospace; fill: #fff; text-anchor: middle; }}
  .helix {{ opacity: .38; }}
  .t1 {{ font: 800 64px 'Segoe UI', Helvetica, Arial, sans-serif; fill: url(#title); text-anchor: middle; }}
  .t2 {{ font: 500 18px 'Segoe UI', Helvetica, Arial, sans-serif; fill: {INK2}; text-anchor: middle; }}
  .glow {{ animation: pulse 3s ease-in-out infinite; }}
  @keyframes pulse {{ 0%,100% {{ opacity: .85 }} 50% {{ opacity: 1 }} }}
</style>
<rect width="{W}" height="{H}" rx="18" fill="url(#bg)"/>
<g class="helix">{''.join(rungs)}
</g>
<rect x="190" y="62" width="520" height="110" rx="16" fill="{BG1}" fill-opacity=".72"/>
<text x="{W/2}" y="128" class="t1 glow">🧬 MotifBench</text>
<text x="{W/2}" y="158" class="t2">hide a motif · hunt it down · score the hunters</text>
</svg>
"""
    _write("hero.svg", svg)


# ---------------------------------------------------------------- scan.svg
def scan():
    rng = np.random.default_rng(7)
    seq = list(rng.choice(list("ACGT"), 44))
    motif, at = "CCATATATAG", 26
    seq[at:at + len(motif)] = list(motif)
    cw, x0, W, H, k = 18, 30, 30 + 44 * 18 + 30, 150, len(motif)
    stops = list(range(0, at + 1, 2)) + [at]           # window start positions
    n = len(stops)
    T = 7.0                                            # seconds per loop
    hold = 0.30                                        # fraction spent on "found"
    frames = []
    for i, s in enumerate(stops):
        pct = i / (n - 1) * (1 - hold) * 100
        frames.append(f"{pct:.1f}% {{ transform: translateX({s * cw}px) }}")
    frames.append(f"100% {{ transform: translateX({at * cw}px) }}")
    found_on = (1 - hold) * 100
    letters = "".join(
        f'<text x="{x0 + i * cw + cw / 2}" y="78" class="b" fill="{BASE_COL[b]}">{b}</text>'
        for i, b in enumerate(seq))
    mot = "".join(
        f'<text x="{x0 + (at + i) * cw + cw / 2}" y="78" class="b" fill="#ffffff">{b}</text>'
        for i, b in enumerate(motif))
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">
<style>
  .b {{ font: 700 17px ui-monospace, Menlo, Consolas, monospace; text-anchor: middle; }}
  .win {{ animation: slide {T}s linear infinite; }}
  @keyframes slide {{ {' '.join(frames)} }}
  .found {{ opacity: 0; animation: show {T}s step-end infinite; }}
  @keyframes show {{ 0% {{ opacity: 0 }} {found_on:.1f}% {{ opacity: 1 }} 100% {{ opacity: 0 }} }}
  .score {{ animation: sc {T}s steps(1) infinite; }}
  .lbl {{ font: 600 14px 'Segoe UI', Helvetica, Arial, sans-serif; fill: {INK2}; }}
  .yay {{ font: 800 16px 'Segoe UI', Helvetica, Arial, sans-serif; fill: #3fe0a8; }}
</style>
<rect width="{W}" height="{H}" rx="14" fill="{BG1}"/>
<text x="{x0}" y="32" class="lbl">sliding a {k}-letter window along the sequence, scoring every position…</text>
{letters}
<g class="found">
  <rect x="{x0 + at * cw}" y="58" width="{k * cw}" height="28" rx="6" fill="#1baf7a"/>
  {mot}
  <text x="{W - 30}" y="118" class="yay" text-anchor="end">✓ best score: CCATATATAG found at position {at}</text>
</g>
<g class="win">
  <rect x="{x0}" y="54" width="{k * cw}" height="36" rx="8" fill="#5fa8ff" fill-opacity=".16"
        stroke="#5fa8ff" stroke-width="2.5"/>
</g>
</svg>
"""
    _write("scan.svg", svg)


# ------------------------------------------------------- algorithms.svg
def algorithms():
    cards = [
        ("🐢", "Greedy", "builds the motif one sequence at a time,\nalways taking the best-looking k-mer", "#2a78d6"),
        ("🎲", "Randomized", "random start, then improve until stuck;\nrestart many times, keep the best", "#eb6834"),
        ("🎰", "Gibbs", "re-roll ONE sequence's site at a time,\nweighted by how well it fits", "#1baf7a"),
    ]
    W, H, cw = 900, 190, 280
    parts = []
    for i, (emo, name, desc, col) in enumerate(cards):
        x = 15 + i * (cw + 15)
        l1, l2 = desc.split("\n")
        parts.append(f"""
  <g class="card" style="animation-delay:{i * 0.5}s">
    <rect x="{x}" y="15" width="{cw}" height="160" rx="16" fill="{BG2}" stroke="{col}" stroke-width="2"/>
    <rect x="{x}" y="15" width="{cw}" height="8" rx="4" fill="{col}"/>
    <text x="{x + 24}" y="72" class="emo">{emo}</text>
    <text x="{x + 76}" y="70" class="name">{name}</text>
    <text x="{x + 24}" y="112" class="desc">{l1}</text>
    <text x="{x + 24}" y="134" class="desc">{l2}</text>
    <rect x="{x + 24}" y="152" width="{cw - 48}" height="6" rx="3" fill="#ffffff" fill-opacity=".08"/>
    <rect x="{x + 24}" y="152" width="{cw - 48}" height="6" rx="3" fill="{col}" class="bar"
          style="animation-delay:{i * 0.5}s"/>
  </g>""")
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">
<style>
  .card {{ animation: bob 3s ease-in-out infinite; }}
  @keyframes bob {{ 0%,100% {{ transform: translateY(0) }} 50% {{ transform: translateY(-5px) }} }}
  .bar {{ transform-box: fill-box; transform-origin: left; animation: load 3s ease-in-out infinite; }}
  @keyframes load {{ 0% {{ transform: scaleX(.05) }} 70%,100% {{ transform: scaleX(1) }} }}
  .emo {{ font: 34px 'Segoe UI Emoji', 'Apple Color Emoji', sans-serif; }}
  .name {{ font: 800 24px 'Segoe UI', Helvetica, Arial, sans-serif; fill: {INK}; }}
  .desc {{ font: 400 13.5px 'Segoe UI', Helvetica, Arial, sans-serif; fill: {INK2}; }}
</style>
{''.join(parts)}
</svg>
"""
    _write("algorithms.svg", svg)


# --------------------------------------------------- logo_strength.gif
def logo_gif():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.textpath import TextPath
    from matplotlib.patches import PathPatch
    from matplotlib.font_manager import FontProperties
    from matplotlib.transforms import Affine2D
    from PIL import Image
    from jaspar_loader import load_motif, get_ppm, JASPAR_DIR, information_content
    from strength import ppm_at_fraction

    mid = "MA0001.3"
    ppm = get_ppm(load_motif(os.path.join(JASPAR_DIR, f"{mid}.jaspar")))
    k = ppm.shape[1]
    fp = FontProperties(family="DejaVu Sans", weight="bold")
    glyph = {b: TextPath((0, 0), b, size=1, prop=fp) for b in "ACGT"}

    def letter(ax, b, x, y, h):
        p = glyph[b]
        ext = p.get_extents()
        t = (Affine2D().translate(-ext.x0, -ext.y0)
             .scale(0.92 / ext.width, h / ext.height).translate(x + 0.04, y))
        ax.add_patch(PathPatch(t.transform_path(p), fc=BASE_COL[b], lw=0))

    fracs = list(np.round(np.linspace(0.2, 1.0, 17), 3))
    seq_fracs = fracs + [1.0] * 6 + fracs[::-1] + [0.2] * 6   # grow, hold, shrink, hold
    cache, frames = {}, []
    for f in seq_fracs:
        if f not in cache:
            P, _, ic = ppm_at_fraction(ppm, f)
            fig, ax = plt.subplots(figsize=(7.2, 2.9), dpi=90, facecolor=BG1)
            ax.set_facecolor(BG1)
            p = np.clip(P, 1e-12, 1)
            col_ic = 2 + (p * np.log2(p)).sum(axis=0)
            for j in range(k):
                y = 0.0
                for bi in np.argsort(P[:, j]):              # smallest at bottom
                    h = P[bi, j] * col_ic[j]
                    if h > 0.01:
                        letter(ax, "ACGT"[bi], j, y, h)
                    y += h
            ax.set_xlim(0, k)
            ax.set_ylim(0, 2)
            ax.set_xticks([])
            ax.set_yticks([0, 1, 2])
            ax.set_ylabel("bits", color=INK2)
            ax.tick_params(colors=INK2)
            for s in ax.spines.values():
                s.set_visible(False)
            ax.set_title(f"{mid}  ·  strength {f:.0%}  ·  information content {ic:.1f} bits",
                         color=INK, fontsize=12, fontweight="bold", loc="left")
            bar = ax.inset_axes([0.0, -0.13, 1.0, 0.04])
            bar.barh(0, f, color="#3fe0a8")
            bar.set_xlim(0, 1)
            bar.axis("off")
            buf = io.BytesIO()
            fig.savefig(buf, format="png", facecolor=BG1, bbox_inches="tight")
            plt.close(fig)
            buf.seek(0)
            cache[f] = Image.open(buf).convert("RGB")
        frames.append(cache[f])
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, "logo_strength.gif")
    frames[0].save(path, save_all=True, append_images=frames[1:],
                   duration=140, loop=0, optimize=True)
    print("wrote", os.path.join("docs", "assets", "logo_strength.gif"),
          f"({os.path.getsize(path) // 1024} KB)")


if __name__ == "__main__":
    hero()
    scan()
    algorithms()
    logo_gif()
