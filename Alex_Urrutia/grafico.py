import glob
import os

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

YEARS  = list(range(2016, 2026))
TOP_N  = 15
SOURCE = "Fuente: SullyGnome — Most Streamed / Most Watched Games on Twitch (2016–2025)"

PALETTE = [
    "#e63946", "#2196f3", "#ff9800", "#4caf50", "#9c27b0",
    "#00bcd4", "#ff5722", "#795548", "#607d8b", "#f06292",
    "#8bc34a", "#673ab7", "#009688", "#ff5252", "#ffc107",
]

def load_total(folder, metric):
    frames = []
    for path in glob.glob(os.path.join(folder, "*.csv")):
        tmp = pd.read_csv(path, header=0)
        tmp.columns = [str(c).strip() for c in tmp.columns]
        frames.append(tmp)
    df = pd.concat(frames, ignore_index=True)
    df["Game"] = df["Game"].astype(str).str.strip()
    df[metric] = pd.to_numeric(df[metric], errors="coerce")
    df = df.dropna(subset=["Game", metric])
    return df.groupby("Game", as_index=False)[metric].sum()

df_s = load_total("Most Streamed", "Stream time (mins)")
df_w = load_total("Most Watched",  "Watch time (mins)")

df_s["stream_rank"] = df_s["Stream time (mins)"].rank(ascending=False, method="min").astype(int)
df_w["watch_rank"]  = df_w["Watch time (mins)"].rank(ascending=False,  method="min").astype(int)

merged = pd.merge(df_s[["Game", "stream_rank"]],
                  df_w[["Game", "watch_rank"]], on="Game", how="inner")

merged = merged[(merged["stream_rank"] <= TOP_N) | (merged["watch_rank"] <= TOP_N)].copy()
merged = merged.sort_values("stream_rank").reset_index(drop=True)

fig, ax = plt.subplots(figsize=(9, 10))
fig.patch.set_facecolor("#fafafa")
ax.set_facecolor("#fafafa")

X_LEFT, X_RIGHT = 0, 1
MIN_GAP = 0.65

color_map = {row["Game"]: PALETTE[i % len(PALETTE)]
             for i, row in merged.iterrows()}

for _, row in merged.iterrows():
    color = color_map[row["Game"]]
    sr, wr = row["stream_rank"], row["watch_rank"]
    lw = 2.0
    alpha = 0.75
    diff = abs(sr - wr)
    if diff >= 5:
        lw, alpha = 2.8, 0.95
    ax.plot([X_LEFT, X_RIGHT], [sr, wr],
            color=color, linewidth=lw, alpha=alpha,
            solid_capstyle="round", zorder=2)
    ax.scatter([X_LEFT, X_RIGHT], [sr, wr],
               color=color, s=70, zorder=3, edgecolors="white", linewidths=1)

left_labels = [(int(row["stream_rank"]), row["Game"]) for _, row in merged.iterrows()]
left_labels.sort(key=lambda x: x[0])
adj_left = []
for rank, game in left_labels:
    if adj_left and rank - adj_left[-1][0] < MIN_GAP:
        rank = adj_left[-1][0] + MIN_GAP
    adj_left.append((rank, game))

for rank, game in adj_left:
    color = color_map[game]
    ax.text(X_LEFT - 0.04, rank, game,
            va="center", ha="right", fontsize=8.5,
            color=color, fontweight="bold")

right_labels = [(int(row["watch_rank"]), row["Game"]) for _, row in merged.iterrows()]
right_labels.sort(key=lambda x: x[0])
adj_right = []
for rank, game in right_labels:
    if adj_right and rank - adj_right[-1][0] < MIN_GAP:
        rank = adj_right[-1][0] + MIN_GAP
    adj_right.append((rank, game))

for rank, game in adj_right:
    color = color_map[game]
    ax.text(X_RIGHT + 0.04, rank, game,
            va="center", ha="left", fontsize=8.5,
            color=color, fontweight="bold")

max_rank = max(
    max(r for r, _ in adj_left),
    max(r for r, _ in adj_right),
    TOP_N
)
ax.set_ylim(max_rank + 0.8, 0.2)
ax.set_xlim(-0.55, 1.55)
ax.axis("off")

for x, label, ha in [(X_LEFT, "Stream Time", "center"), (X_RIGHT, "Watch Time", "center")]:
    ax.axvline(x, color="#cccccc", linewidth=1.5, zorder=0)
    ax.text(x, 0, label, ha=ha, va="bottom", fontsize=12,
            fontweight="bold", color="#444444")

all_ranks = set(int(r["stream_rank"]) for _, r in merged.iterrows()) | \
            set(int(r["watch_rank"])  for _, r in merged.iterrows())
for r in sorted(all_ranks):
    if r <= max_rank + 0.5:
        ax.text(X_LEFT  - 0.015, r, f"#{r}", ha="right", va="center",
                fontsize=7.5, color="#aaaaaa")
        ax.text(X_RIGHT + 0.015, r, f"#{r}", ha="left",  va="center",
                fontsize=7.5, color="#aaaaaa")

ax.set_title(
    "Slope Chart — Stream Rank vs Watch Rank\n"
    "Top juegos en Twitch, acumulado 2016–2025",
    fontsize=13, fontweight="bold", pad=18, loc="center",
)
fig.text(0.5, -0.01, SOURCE, ha="center", fontsize=8, color="gray", style="italic")

fig.text(0.5, 0.01,
         "Línea hacia abajo → más visto de lo esperado (alto engagement)   "
         "Línea hacia arriba → más streameado que visto",
         ha="center", fontsize=8, color="#777777", style="italic")

plt.tight_layout()
plt.savefig("Grafico Streamed Vs Watched.png", dpi=150, bbox_inches="tight")
plt.show()


