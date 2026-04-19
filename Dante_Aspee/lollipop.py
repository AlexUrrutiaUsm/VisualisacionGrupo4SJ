import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches
import os

# 1. DIRECTORIO Y ESTILO
os.makedirs("Dante_Aspee", exist_ok=True)

COLOR_MAIN   = "#6441A5" # Morado (Audiencia)
COLOR_ACCENT = "#F0A500" # Naranja (Ingresos)
COLOR_BG     = "#0F0F13"
COLOR_PANEL  = "#1A1A24"
COLOR_TEXT   = "#EFEFF1"
COLOR_MUTED  = "#D1D1D1"

plt.rcParams.update({
    "figure.facecolor": COLOR_BG,
    "axes.facecolor":   COLOR_PANEL,
    "axes.edgecolor":   COLOR_MUTED,
    "xtick.color":      COLOR_MUTED,
    "ytick.color":      COLOR_MUTED,
    "text.color":       COLOR_TEXT,
    "font.family":      "DejaVu Sans",
})

# 2. CARGA DE DATOS DESDE CSV
csv_path = "data/twitch_data.csv"

try:
    df = pd.read_csv(csv_path)
    years   = df['anio'].values
    revenue = df['ingresos'].values
    viewers = df['viewers'].values
except FileNotFoundError:
    print(f"Error: No se encontró el archivo en {csv_path}")

# 3. CREACIÓN DE LA GRÁFICA
fig, ax = plt.subplots(figsize=(12, 11), facecolor=COLOR_BG)

# Tallos (Ingresos)
ax.vlines(years, 0, revenue, color=COLOR_ACCENT, linewidth=2, alpha=0.6, zorder=1)

# Hexágonos (Audiencia)
sizes = viewers / 0.8 
ax.scatter(years, revenue, s=sizes, marker='H', color=COLOR_MAIN, 
            edgecolors=COLOR_TEXT, linewidths=1.5, zorder=5)

# 4. ETIQUETAS DE DATOS
for yr, rv, vk in zip(years, revenue, viewers):
    ax.text(yr, rv + 0.18, f"${rv:.2f}B", ha="center", fontsize=11, fontweight='bold', color=COLOR_ACCENT)
    v_str = f"{vk/1000:.1f}M" if vk >= 1000 else f"{vk}K"
    ax.text(yr, rv, v_str, ha="center", va="center", fontsize=9, color="white", fontweight='black', zorder=10)

# 5. LEYENDA
orange_patch = mpatches.Patch(color=COLOR_ACCENT, label='Ingresos ($ Billones)')
purple_patch = mpatches.Patch(color=COLOR_MAIN, label='Audiencia (Promedio Viewers)')
ax.legend(handles=[orange_patch, purple_patch], loc='upper left', 
          facecolor=COLOR_PANEL, edgecolor=COLOR_MUTED, fontsize=10)

# 6. CONFIGURACIÓN FINAL
ax.set_title("Criterio: Desempeño Financiero y Audiencia de Twitch (2016-2024)", 
             loc='left', pad=35, fontsize=16, fontweight='bold')
ax.set_ylabel("Ingresos (Miles de Millones USD)", color=COLOR_MUTED, fontsize=11, labelpad=15)
ax.set_ylim(0, 2.6)
ax.set_xticks(years)
ax.set_xticklabels([str(int(y)) for y in years]) # Aseguramos que el año sea entero
ax.tick_params(axis='x', pad=15)
ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("$%.1fB"))

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_color(COLOR_MUTED)
ax.spines['left'].set_color(COLOR_MUTED)
ax.grid(axis='y', alpha=0.1, linestyle='--')

# 7. FUENTES
fuentes_text = "Fuentes: TwitchTracker | Business of Apps (2026) | DemandSage (2026)"
fig.text(0.5, 0.02, fuentes_text, ha="center", fontsize=7.5, color=COLOR_MUTED)

plt.savefig("Dante_Aspee/sLollipop_Twitch.png.png", dpi=200, bbox_inches="tight", facecolor=COLOR_BG)
plt.show()