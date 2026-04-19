import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("Most watched games on Twitch - SullyGnome.csv", header=0)
df.columns = [str(c).strip() for c in df.columns]
cols = ["Game", "Watch time (mins)", "Stream time (mins)", "Peak viewers", "Average viewers", "Streamers"]
df = df[cols].copy()
df = df.dropna(subset=["Game"])
df["Game"] = df["Game"].astype(str)
for c in cols[1:]:
    df[c] = pd.to_numeric(df[c], errors="coerce")
df = df.dropna()

top8 = df.nlargest(8, "Watch time (mins)").reset_index(drop=True)

SOURCE = "Fuente: SullyGnome — Most Watched Games on Twitch"

metrics_r = ["Watch time (mins)", "Stream time (mins)", "Peak viewers", "Average viewers", "Streamers"]
mlabels_r = ["Watch Time", "Stream Time", "Peak Viewers", "Avg Viewers", "Streamers"]
N = len(metrics_r)
angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
angles += angles[:1]

fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
fig.suptitle("Radar: Métricas Normalizadas — Most Watched Games on Twitch (Top 8)",
             fontsize=14, fontweight="bold", y=0.98)

ax.set_xticks(angles[:-1])
ax.set_xticklabels(mlabels_r, fontsize=11)
ax.set_yticks([0.25, 0.5, 0.75, 1.0])
ax.set_yticklabels(["0.25", "0.50", "0.75", "1.0"], fontsize=8)
ax.set_ylim(0, 1)
ax.set_theta_offset(np.pi / 2)
ax.set_theta_direction(-1)
ax.grid(color="gray", linestyle="--", alpha=0.4)

colors_r = plt.cm.Set2(np.linspace(0, 0.9, len(top8)))
for i, (_, row) in enumerate(top8.iterrows()):
    vs = [row[m] / df[m].max() for m in metrics_r]
    vs += vs[:1]
    ax.plot(angles, vs, color=colors_r[i], linewidth=2, label=row["Game"])
    ax.fill(angles, vs, color=colors_r[i], alpha=0.07)

ax.legend(loc="upper right", bbox_to_anchor=(1.45, 1.15), fontsize=10)
fig.text(0.5, 0.01, SOURCE, ha="center", fontsize=8, color="gray", style="italic")

plt.tight_layout(rect=[0, 0.03, 1, 0.96])
plt.savefig("grafico_most_watched.png", dpi=150, bbox_inches="tight")
plt.show()
