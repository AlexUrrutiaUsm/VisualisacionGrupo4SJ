import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches
import os

# 1. DIRECTORIO Y ESTILO
os.makedirs("Dante_Aspee", exist_ok=True)

COLOR_MAIN   = "#6441A5" 
COLOR_ACCENT = "#F0A500" 
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

# 2. DATOS (2016-2024)
years   = np.array([2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024])
revenue = np.array([0.275, 0.300, 0.880, 1.23, 1.89, 2.05, 1.90, 1.96, 1.80])
viewers = np.array([757, 1100, 1500, 1260, 2120, 2780, 2580, 2440, 2370])

# 3. CREACIÓN DE LA GRÁFICA
fig, ax = plt.subplots(figsize=(12, 11), facecolor=COLOR_BG) 

# Tallos en NARANJA (Representan Ingresos)
ax.vlines(years, 0, revenue, color=COLOR_ACCENT, linewidth=2, alpha=0.6, zorder=1)

# Hexágonos en MORADO (Representan Audiencia)
sizes = viewers / 0.8 
ax.scatter(years, revenue, s=sizes, marker='H', color=COLOR_MAIN, 
            edgecolors=COLOR_TEXT, linewidths=1.5, zorder=5)

# 4. ETIQUETAS DE DATOS
for yr, rv, vk in zip(years, revenue, viewers):
    # Texto de Ingresos arriba (en Naranja)
    ax.text(yr, rv + 0.18, f"${rv:.2f}B", ha="center", fontsize=11, fontweight='bold', color=COLOR_ACCENT)
    
    # Texto de Audiencia ADENTRO (en Blanco)
    v_str = f"{vk/1000:.1f}M" if vk >= 1000 else f"{vk}K"
    ax.text(yr, rv, v_str, ha="center", va="center", fontsize=9, color="white", fontweight='black', zorder=10)

# 5. LEYENDA SIMPLE
orange_patch = mpatches.Patch(color=COLOR_ACCENT, label='Ingresos ($ Billones)')
purple_patch = mpatches.Patch(color=COLOR_MAIN, label='Audiencia (Promedio Viewers)')
ax.legend(handles=[orange_patch, purple_patch], loc='upper left', 
          facecolor=COLOR_PANEL, edgecolor=COLOR_MUTED, fontsize=10)

# 6. CONFIGURACIÓN FINAL DE EJES
ax.set_title("Criterios: Desempeño Financiero y Audiencia de Twitch (2016-2024)", 
             loc='left', pad=35, fontsize=16, fontweight='bold')
ax.set_ylabel("Ingresos (Miles de Millones USD)", color=COLOR_MUTED, fontsize=11, labelpad=15)
ax.set_ylim(0, 2.6)
ax.set_xticks(years)
ax.set_xticklabels([str(y) for y in years])
ax.tick_params(axis='x', pad=15)
ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("$%.1fB"))

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_color(COLOR_MUTED)
ax.spines['left'].set_color(COLOR_MUTED)
ax.grid(axis='y', alpha=0.1, linestyle='--')

# 7. FUENTES 
fuentes_text = "Fuentes: TwitchTracker · twitchtracker.com/statistics | Business of Apps (2026) · businessofapps.com/data/twitch-statistics | DemandSage (2026) · demandsage.com/twitch-users"
fig.text(0.5, 0.02, fuentes_text, ha="center", fontsize=7.5, color=COLOR_MUTED)

# Guardado final
plt.savefig("Dante_Aspee/Lollipop_Twitch.png", dpi=200, bbox_inches="tight", facecolor=COLOR_BG)
plt.show()