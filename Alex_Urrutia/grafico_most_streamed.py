import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches

df = pd.read_csv("Most streamed games on Twitch - SullyGnome.csv", header=0)
df.columns = [str(c).strip() for c in df.columns]
cols = ["Game", "Watch time (mins)", "Stream time (mins)", "Peak viewers", "Average viewers", "Streamers"]
df = df[cols].copy()
df = df.dropna(subset=["Game"])
df["Game"] = df["Game"].astype(str)
for c in cols[1:]:
    df[c] = pd.to_numeric(df[c], errors="coerce")
df = df.dropna()

top8 = df.nlargest(8, "Stream time (mins)").reset_index(drop=True)

SOURCE = "Fuente: SullyGnome — Most Streamed Games on Twitch"

total_w = top8["Stream time (mins)"].sum()
shares_raw = top8["Stream time (mins)"].values / total_w * 100
shares_int = np.floor(shares_raw).astype(int)
remainder = 100 - shares_int.sum()
if remainder > 0:
    frac = shares_raw - shares_int
    idxs = np.argsort(frac)[::-1][:remainder]
    shares_int[idxs] += 1
flat = np.repeat(np.arange(len(top8)), shares_int)
grid = np.flipud(flat.reshape(10, 10))
cw = list(plt.cm.tab10.colors[:len(top8)])
cmap_w = mcolors.ListedColormap(cw)

fig, ax = plt.subplots(figsize=(10, 9))
fig.suptitle("Waffle: Proporción del Stream Time Total\nMost Streamed Games on Twitch (Top 8)",
             fontsize=14, fontweight="bold", y=0.98)

ax.pcolormesh(grid, cmap=cmap_w, vmin=-0.5, vmax=len(top8) - 0.5,
              edgecolors="white", linewidths=3)
ax.set_aspect("equal")
ax.set_xticks([])
ax.set_yticks([])
for spine in ax.spines.values():
    spine.set_visible(False)

patches = [mpatches.Patch(color=cw[i],
           label=f"{top8.loc[i, 'Game']} ({shares_raw[i]:.1f}%)")
           for i in range(len(top8))]
ax.legend(handles=patches, loc="lower center",
          bbox_to_anchor=(0.5, -0.12), ncol=2, fontsize=10, frameon=True)
fig.text(0.5, -0.02, SOURCE, ha="center", fontsize=8, color="gray", style="italic",
         transform=ax.transAxes)

plt.tight_layout(rect=[0, 0.08, 1, 0.96])
plt.savefig("grafico_most_streamed.png", dpi=150, bbox_inches="tight")
plt.show()
