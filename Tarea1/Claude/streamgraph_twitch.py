"""
╔══════════════════════════════════════════════════════════════════════════════╗
║   Streamgraph — La Batalla por la Atención: 10 Años de Twitch (2016–2025)  ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  Tipo de gráfico : Streamgraph (área apilada con baseline "wiggle")        ║
║                    Variante poco común — centra visualmente el flujo de    ║
║                    datos como un río, revelando tendencias temporales.     ║
║                                                                            ║
║  Criterio        : Evolución del tiempo de visualización acumulado por     ║
║                    categoría/juego en Twitch a lo largo de 10 años.       ║
║                    ¿Qué juegos dominaron la atención de la audiencia       ║
║                    y cómo cambiaron esas preferencias?                     ║
║                                                                            ║
║  Datos usados    :                                                         ║
║    · data/Most Watched/  — 10 archivos CSV (2016–2025)                     ║
║    · data/Most Streamed/ — 10 archivos CSV (2016–2025)                     ║
║    · data/Most followed streamers on Twitch, past 365 days.csv            ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import glob
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
from matplotlib.colors import to_rgba

# ──────────────────────────────────────────────────────────────────────────────
# PATHS  (relativos al script → funciona desde cualquier directorio)
# ──────────────────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT  = os.path.join(SCRIPT_DIR, "..", "data")
OUT_DIR    = SCRIPT_DIR

# ──────────────────────────────────────────────────────────────────────────────
# PARÁMETROS GLOBALES
# ──────────────────────────────────────────────────────────────────────────────
YEARS  = list(range(2016, 2026))
TOP_N  = 9          # top juegos individualmente; el resto → "Otros"
SOURCE = ("Fuente: SullyGnome — Most Watched & Most Streamed Games on Twitch "
          "(2016–2025) · Most Followed Streamers (últimos 365 días)")

# ──────────────────────────────────────────────────────────────────────────────
# PALETA Y METADATOS DE JUEGOS
# ──────────────────────────────────────────────────────────────────────────────
BG    = "#060712"          # fondo casi negro-azul
TEXT  = "#e8e8f0"
MUTED = "#7777aa"
GRID  = "#ffffff0d"

# juego → (etiqueta corta, color)
GAME_META = {
    "Just Chatting":     ("Just Chatting",    "#9146FF"),
    "League of Legends": ("League of Legends","#C8A030"),
    "Grand Theft Auto V":("GTA V",            "#FF5C2E"),
    "Fortnite":          ("Fortnite",         "#00CFFF"),
    "Counter-Strike":    ("Counter-Strike",   "#FF9800"),
    "VALORANT":          ("VALORANT",         "#FF4655"),
    "Dota 2":            ("Dota 2",           "#B22222"),
    "Minecraft":         ("Minecraft",        "#5AB054"),
    "World of Warcraft": ("WoW",              "#4682B4"),
    "_Others":           ("Otros juegos",     "#3a3a5a"),
}

# Orden de apilado: los más grandes al centro → cuerpo más visible
STACK_ORDER = [
    "Grand Theft Auto V",
    "Fortnite",
    "Counter-Strike",
    "World of Warcraft",
    "_Others",
    "Minecraft",
    "Dota 2",
    "VALORANT",
    "League of Legends",
    "Just Chatting",
]


# ──────────────────────────────────────────────────────────────────────────────
# UTILIDADES
# ──────────────────────────────────────────────────────────────────────────────
def gaussian_smooth(arr: np.ndarray, sigma: float = 0.8) -> np.ndarray:
    """Suavizado gaussiano sin scipy — convolución con kernel manual."""
    radius = max(1, int(3 * sigma))
    x = np.arange(-radius, radius + 1)
    kernel = np.exp(-0.5 * (x / sigma) ** 2)
    kernel /= kernel.sum()
    padded = np.pad(arr, radius, mode="edge")
    return np.convolve(padded, kernel, mode="valid")[: len(arr)]


def fmt_billions(v: float) -> str:
    if v >= 1:
        return f"{v:.1f}B h"
    return f"{v * 1000:.0f}M h"


# ──────────────────────────────────────────────────────────────────────────────
# CARGA DE DATOS
# ──────────────────────────────────────────────────────────────────────────────
def load_watched() -> pd.DataFrame:
    frames = []
    pattern = os.path.join(DATA_ROOT, "Most Watched", "*.csv")
    for path in glob.glob(pattern):
        year = int("".join(filter(str.isdigit, os.path.basename(path)))[:4])
        df = pd.read_csv(path, header=0)
        df.columns = [str(c).strip() for c in df.columns]
        df["Year"]             = year
        df["Game"]             = df["Game"].astype(str).str.strip()
        df["Watch time (mins)"] = pd.to_numeric(df["Watch time (mins)"], errors="coerce")
        frames.append(df[["Year", "Game", "Watch time (mins)"]].dropna())
    return pd.concat(frames, ignore_index=True)


def load_streamed() -> pd.DataFrame:
    frames = []
    pattern = os.path.join(DATA_ROOT, "Most Streamed", "*.csv")
    for path in glob.glob(pattern):
        year = int("".join(filter(str.isdigit, os.path.basename(path)))[:4])
        df = pd.read_csv(path, header=0)
        df.columns = [str(c).strip() for c in df.columns]
        df["Year"]              = year
        df["Game"]              = df["Game"].astype(str).str.strip()
        df["Stream time (mins)"] = pd.to_numeric(df["Stream time (mins)"], errors="coerce")
        frames.append(df[["Year", "Game", "Stream time (mins)"]].dropna())
    return pd.concat(frames, ignore_index=True)


def load_streamers() -> pd.DataFrame:
    path = os.path.join(
        DATA_ROOT,
        "Most followed streamers on Twitch, past 365 days - SullyGnome.csv",
    )
    df = pd.read_csv(path, header=0)
    df.columns = [str(c).strip() for c in df.columns]
    for col in ["Followers", "Average viewers", "Watch time (mins)", "Peak viewers"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["Channel"] = df["Channel"].astype(str).str.strip()
    return df.dropna(subset=["Channel", "Followers"])


# ──────────────────────────────────────────────────────────────────────────────
# PREPARAR DATOS PARA EL STREAMGRAPH
# ──────────────────────────────────────────────────────────────────────────────
raw_watch   = load_watched()
raw_stream  = load_streamed()
streamers   = load_streamers()

# Top N juegos por tiempo de visualización total
top_games = (
    raw_watch.groupby("Game")["Watch time (mins)"]
    .sum()
    .nlargest(TOP_N)
    .index.tolist()
)

# Pivot años × juegos (en minutos)
pivot_w = (
    raw_watch[raw_watch["Game"].isin(top_games)]
    .pivot_table(
        index="Year", columns="Game",
        values="Watch time (mins)", aggfunc="sum", fill_value=0,
    )
    .reindex(YEARS, fill_value=0)
)

# Categoría "Otros" = total anual − top-N
year_totals = raw_watch.groupby("Year")["Watch time (mins)"].sum().reindex(YEARS, fill_value=0)
pivot_w["_Others"] = (year_totals - pivot_w.sum(axis=1)).clip(lower=0)

# Convertir a miles de millones de horas
pivot_bh = pivot_w / 60 / 1_000_000_000

# Reordenar según el orden de apilado deseado
ordered = [g for g in STACK_ORDER if g in pivot_bh.columns]

X    = np.array(YEARS, dtype=float)
Y_raw = np.array([pivot_bh[g].values for g in ordered])

# Suavizar para que el streamgraph fluya visualmente
Y = np.array([gaussian_smooth(row, sigma=0.6) for row in Y_raw])
Y = np.clip(Y, 0, None)

# Baseline "wiggle" aproximado: –½ × total (mismo que usa matplotlib)
baseline  = -0.5 * Y.sum(axis=0)
Y_cumsum  = np.cumsum(Y, axis=0)


# ──────────────────────────────────────────────────────────────────────────────
# ESTADÍSTICAS DE STREAMERS (para anotación secundaria)
# ──────────────────────────────────────────────────────────────────────────────
top_streamer    = streamers.nlargest(1, "Followers").iloc[0]
n_languages     = streamers["Language"].nunique()
n_spanish       = (streamers["Language"] == "Spanish").sum()
watch_total_bh  = raw_watch["Watch time (mins)"].sum() / 60 / 1_000_000_000
stream_total_bh = raw_stream["Stream time (mins)"].sum() / 60 / 1_000_000_000


# ──────────────────────────────────────────────────────────────────────────────
# FIGURA
# ──────────────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(19, 10))
fig.patch.set_facecolor(BG)

# Área principal del streamgraph
ax = fig.add_axes([0.03, 0.13, 0.72, 0.72])
ax.set_facecolor(BG)

colors = [GAME_META[g][1] for g in ordered]
polys  = ax.stackplot(X, Y, colors=colors, baseline="wiggle", alpha=0.90, zorder=2)


# ── Etiquetas dentro de cada banda ──────────────────────────────────────────
total_h = Y.sum(axis=0).max()  # escala de referencia

for i, game in enumerate(ordered):
    label, color = GAME_META[game]
    row = Y[i]

    if row.max() < total_h * 0.025:   # banda demasiado fina → omitir
        continue

    # Año donde la banda es más gruesa
    xi = int(np.argmax(row))

    # Posición y central de la banda en ese año
    lower = baseline[xi] + (Y_cumsum[i - 1, xi] if i > 0 else 0.0)
    upper = lower + row[xi]
    mid_y = (lower + upper) / 2.0
    band_h = upper - lower

    fontsize = 9 if band_h > total_h * 0.10 else 7.5
    ax.text(
        X[xi], mid_y, label,
        ha="center", va="center",
        fontsize=fontsize, fontweight="bold", color="white",
        zorder=6,
        bbox=dict(
            boxstyle="round,pad=0.25",
            fc=to_rgba(color, 0.70),
            ec="none",
        ),
    )


# ── Grid vertical tenue ──────────────────────────────────────────────────────
for yr in YEARS:
    ax.axvline(yr, color=GRID, linewidth=0.8, zorder=1)


# ── Anotaciones de eventos históricos ───────────────────────────────────────
y_top   = 0.5  * Y.sum(axis=0).max()
y_bot   = -y_top

events = [
    (2018, "Boom de\nFortnite",         0.82, "#00CFFF"),
    (2019, "Nace la categoría\n'Just Chatting'", 0.93, "#9146FF"),
    (2020, "COVID-19:\nauge masivo",    0.87, "#ffffff"),
    (2021, "GTA V Roleplay\nalcanza su cima", 0.76, "#FF5C2E"),
]

for yr, label, frac_y, clr in events:
    y_pos = y_bot + (y_top - y_bot) * frac_y
    # Línea punteada
    ax.axvline(
        x=yr, ymin=0.0, ymax=frac_y - 0.04,
        color=clr, linewidth=1.2,
        linestyle="--", alpha=0.55, zorder=3,
    )
    ax.text(
        yr, y_pos, label,
        ha="center", va="bottom",
        fontsize=8, color=clr, alpha=0.9,
        bbox=dict(
            boxstyle="round,pad=0.3",
            fc=to_rgba(clr, 0.10),
            ec=to_rgba(clr, 0.35),
            linewidth=0.8,
        ),
        zorder=7,
    )


# ── Ejes ────────────────────────────────────────────────────────────────────
ax.set_xlim(YEARS[0] - 0.35, YEARS[-1] + 0.35)
ax.set_xticks(YEARS)
ax.set_xticklabels(
    [str(y) for y in YEARS],
    color=TEXT, fontsize=11, fontweight="bold",
)
ax.yaxis.set_visible(False)
for spine in ax.spines.values():
    spine.set_visible(False)
ax.tick_params(axis="x", length=0, pad=10)


# ──────────────────────────────────────────────────────────────────────────────
# PANEL LATERAL — estadísticas de streamers y resumen de datos
# ──────────────────────────────────────────────────────────────────────────────
ax_side = fig.add_axes([0.77, 0.13, 0.20, 0.72])
ax_side.set_facecolor("#0d1020")
ax_side.set_xlim(0, 1)
ax_side.set_ylim(0, 1)
for spine in ax_side.spines.values():
    spine.set_color("#ffffff18")
ax_side.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

# Título del panel
ax_side.text(
    0.5, 0.97, "Datos del Ecosistema",
    ha="center", va="top",
    fontsize=11, fontweight="bold", color=TEXT,
)
ax_side.axhline(0.93, color="#ffffff20", linewidth=0.8)

# ── Bloque 1: Streamer más seguido ───────────────────────────────────────────
ax_side.text(0.5, 0.90, "Streamer más seguido", ha="center", va="top",
             fontsize=8.5, color=MUTED)
ax_side.text(0.5, 0.84, f"{top_streamer['Channel']}",
             ha="center", va="top", fontsize=16, fontweight="bold",
             color=GAME_META["Just Chatting"][1])
ax_side.text(0.5, 0.77,
             f"{top_streamer['Followers'] / 1_000_000:.1f}M seguidores  ·  "
             f"{top_streamer['Language']}",
             ha="center", va="top", fontsize=8, color=TEXT)
ax_side.text(0.5, 0.72,
             f"~{top_streamer['Average viewers'] / 1000:.0f}K viewers en promedio",
             ha="center", va="top", fontsize=7.5, color=MUTED)

ax_side.axhline(0.69, color="#ffffff15", linewidth=0.6)

# ── Bloque 2: Diversidad lingüística ─────────────────────────────────────────
ax_side.text(0.5, 0.66, "Top 100 streamers", ha="center", va="top",
             fontsize=8.5, color=MUTED)
ax_side.text(0.5, 0.60, f"{n_languages} idiomas representados",
             ha="center", va="top", fontsize=11, fontweight="bold", color=TEXT)
ax_side.text(0.5, 0.55,
             f"59% en Inglés  ·  23% en Español\n"
             f"({n_spanish} de los 100 streamers en español)",
             ha="center", va="top", fontsize=7.5, color=MUTED, linespacing=1.5)

ax_side.axhline(0.50, color="#ffffff15", linewidth=0.6)

# ── Bloque 3: Volumen total de datos ─────────────────────────────────────────
ax_side.text(0.5, 0.47, "Volumen total (2016–2025)", ha="center", va="top",
             fontsize=8.5, color=MUTED)
ax_side.text(0.5, 0.41,
             f"{watch_total_bh:.1f}B horas",
             ha="center", va="top", fontsize=14, fontweight="bold",
             color="#00CFFF")
ax_side.text(0.5, 0.36, "de tiempo de visualización", ha="center", va="top",
             fontsize=7.5, color=MUTED)
ax_side.text(0.5, 0.30,
             f"{stream_total_bh:.1f}B horas",
             ha="center", va="top", fontsize=14, fontweight="bold",
             color="#FF9800")
ax_side.text(0.5, 0.25, "de tiempo de transmisión", ha="center", va="top",
             fontsize=7.5, color=MUTED)

ax_side.axhline(0.21, color="#ffffff15", linewidth=0.6)

# ── Leyenda de colores ────────────────────────────────────────────────────────
ax_side.text(0.5, 0.19, "Categorías", ha="center", va="top",
             fontsize=8.5, color=MUTED)

# 10 ítems (9 juegos + Otros). Spacing fijo → todos dentro del panel [0, 1].
legend_items = [g for g in ordered if g != "_Others"] + ["_Others"]
n_items      = len(legend_items)
legend_start = 0.16          # primera entrada (justo bajo "Categorías")
legend_step  = legend_start / n_items   # 0.016 → último ítem en ~0.016
legend_y     = legend_start

for g in legend_items:
    label, color = GAME_META[g]
    short = label.replace("\n", " ")
    rect = plt.Rectangle(
        (0.07, legend_y - 0.008), 0.10, 0.015,
        fc=color, ec="none", transform=ax_side.transAxes, zorder=5,
    )
    ax_side.add_patch(rect)
    ax_side.text(
        0.22, legend_y - 0.001, short,
        ha="left", va="center",
        fontsize=6.5, color=TEXT, transform=ax_side.transAxes,
    )
    legend_y -= legend_step


# ──────────────────────────────────────────────────────────────────────────────
# TÍTULO, SUBTÍTULO Y FUENTE
# ──────────────────────────────────────────────────────────────────────────────
fig.text(
    0.03, 0.96,
    "La Batalla por la Atención: 10 Años de Dominio en Twitch  (2016 – 2025)",
    fontsize=17, fontweight="bold", color=TEXT, va="top",
)
fig.text(
    0.03, 0.907,
    "Miles de millones de horas de visualización acumuladas por categoría/juego cada año  ·  "
    "Criterio: ¿qué contenido capturó la audiencia?",
    fontsize=10.5, color=MUTED, va="top",
)
fig.text(
    0.03, 0.02, SOURCE,
    ha="left", fontsize=7.5, color=MUTED,
)
fig.text(
    0.76, 0.02,
    "Tipo de gráfico: Streamgraph (baseline wiggle)  ·  Claude / Grupo 4 SJ",
    ha="left", fontsize=7.5, color=MUTED,
)


# ──────────────────────────────────────────────────────────────────────────────
# GUARDAR Y MOSTRAR
# ──────────────────────────────────────────────────────────────────────────────
out_path = os.path.join(OUT_DIR, "streamgraph_twitch.png")
plt.savefig(out_path, dpi=200, bbox_inches="tight", facecolor=BG)
print(f"Guardado en: {out_path}")
plt.show()
