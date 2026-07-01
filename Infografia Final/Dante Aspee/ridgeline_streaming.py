import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager
from scipy.stats import gaussian_kde

# ----------------------------------------------------------------------
# 1. CARGA Y PREPARACIÓN DE DATOS
# ----------------------------------------------------------------------

CSV_PATH = r"Infografia Final/Dante Aspee/Encuesta sobre Hábitos de Consumo de Streaming y Visualización de Datos.csv"

COL_PLAT = "¿Cuál de las siguientes plataformas de streaming utilizas con más frecuencia?"
COL_HORAS = "En promedio, ¿cuántas horas dedicas a consumir contenido en plataformas de streaming a la semana?"

# Punto medio representativo de cada rango de horas (variable categórica -> continua)
HORAS_MAP = {
    "Menos de 3 horas": 1.5,
    "3 a 7 horas": 5.0,
    "8 a 15 horas": 11.5,
    "16 a 25 horas": 20.5,
    "Más de 25 horas": 28.0,
}

# Precio mensual de suscripción individual estándar, CLP (2026) - mercado chileno,
# coherente con el país donde se aplicó la encuesta.
# Fuentes: spotify.com/cl/premium (Spotify); La Tercera / Comparaiso.cl (Netflix Estándar);
# Meganoticias / La Hora (YouTube Premium individual, tras alza de abril 2025, vigente en 2026).
PRECIOS_CLP = {
    "Netflix": 9990,                                       # Plan Estándar sin publicidad
    "Spotify / Apple Music (Streaming de audio)": 4950,    # Spotify Premium Individual
    "YouTube Premium": 5500,                               # Plan Individual
}
PRECIOS_USD = PRECIOS_CLP  # alias para no romper referencias posteriores en el script

df = pd.read_csv(CSV_PATH)
sub = df[[COL_PLAT, COL_HORAS]].dropna().copy()
sub = sub[~sub[COL_PLAT].isin(["No utilizo plataformas de streaming", "Otra/Varias (Especificar)"])]
sub["horas_num"] = sub[COL_HORAS].map(HORAS_MAP)

# Nos quedamos con las plataformas que tienen suficiente n para estimar densidad (>=5 respuestas)
plataformas_validas = [p for p in PRECIOS_USD if (sub[COL_PLAT] == p).sum() >= 5]
sub = sub[sub[COL_PLAT].isin(plataformas_validas)]

# Orden de las montañas: de menor a mayor precio (de abajo hacia arriba)
orden = sorted(plataformas_validas, key=lambda p: PRECIOS_USD[p])

NOMBRES_CORTOS = {
    "Spotify / Apple Music (Streaming de audio)": "Spotify / Apple Music",
    "Netflix": "Netflix",
    "YouTube Premium": "YouTube Premium",
}

# ----------------------------------------------------------------------
# 2. ESTILO VISUAL
# ----------------------------------------------------------------------

plt.rcParams["font.family"] = "DejaVu Sans"
BG = "#ffffff"
FG = "#1a1a1a"
GRID = "#cccccc"

colores = {
    "Netflix": "#E50914",
    "Spotify / Apple Music (Streaming de audio)": "#1DB954",
    "YouTube Premium": "#3EA6FF",
}

fig, ax = plt.subplots(figsize=(11, 7.5), facecolor=BG)
ax.set_facecolor(BG)

x_grid = np.linspace(-2, 32, 500)
overlap = 1.6  # cuánto se superponen las montañas verticalmente

y_ticks = []
y_labels = []

for i, plat in enumerate(orden):
    horas = sub.loc[sub[COL_PLAT] == plat, "horas_num"].values
    n = len(horas)
    color = colores[plat]
    base_y = i * overlap

    # Densidad KDE (con jitter mínimo para evitar singularidades si hay pocos valores únicos)
    kde = gaussian_kde(horas, bw_method=0.4)
    density = kde(x_grid)
    density = density / density.max() * 1.35  # normalizar altura de cada montaña

    curve_y = base_y + density

    ax.fill_between(x_grid, base_y, curve_y, color=color, alpha=0.55, zorder=10 - i, linewidth=0)
    ax.plot(x_grid, curve_y, color=color, linewidth=2, zorder=10 - i)

    # Línea base de cada categoría
    ax.axhline(base_y, color=GRID, linewidth=0.8, zorder=1, xmin=0.0)

    # Etiqueta de plataforma + precio (ambas ancladas a la línea base de SU propia montaña)
    precio = PRECIOS_USD[plat]
    label = f"{NOMBRES_CORTOS[plat]}"
    ax.text(-3.2, base_y + 0.42, label, color=FG, fontsize=12.5, fontweight="bold",
            ha="right", va="bottom")
    ax.text(-3.2, base_y + 0.08, f"${precio:,.0f} CLP/mes  ·  n={n}".replace(",", "."), color="#666666",
            fontsize=9.5, ha="right", va="bottom")

    y_ticks.append(base_y)
    y_labels.append("")

# Eje X
ax.set_xlim(-9, 32)
ax.set_ylim(-0.8, (len(orden) - 1) * overlap + 1.8)
ax.set_xticks([1.5, 5, 11.5, 20.5, 28])
ax.set_xticklabels(["<3h", "3–7h", "8–15h", "16–25h", ">25h"], color=FG, fontsize=10.5)
ax.set_xlabel("Horas semanales dedicadas a la plataforma", color=FG, fontsize=11, labelpad=12)

ax.set_yticks([])
for spine in ["top", "right", "left"]:
    ax.spines[spine].set_visible(False)
ax.spines["bottom"].set_color("#999999")
ax.tick_params(axis="x", colors=FG)

ax.grid(axis="x", color=GRID, linewidth=0.5, alpha=0.5, zorder=0)

# Título y subtítulo
fig.text(0.5, 0.965, "¿Pagar más significa consumir más?", color=FG,
          fontsize=18, fontweight="bold", ha="center")
fig.text(0.5, 0.935,
         "Distribución de horas semanales de consumo según plataforma de streaming, vs. precio de suscripción (CLP, 2026)",
         color="#444444", fontsize=10.5, ha="center")

# Fuente
fig.text(0.5, 0.02,
         "Fuente: Encuesta propia \"Hábitos de Consumo de Streaming\" (n=29, Chile, 2026) · "
         "Precios oficiales Chile: Spotify, Netflix, YouTube Premium (2026)",
         color="#888888", fontsize=8.5, ha="center")

plt.tight_layout(rect=[0, 0.04, 1, 0.92])
plt.savefig("ridgeline_streaming.png", dpi=200, facecolor=BG)
print("Listo. Guardado en ridgeline_streaming.png")

# Resumen estadístico para la ficha técnica / conclusiones
print("\n--- Resumen estadístico ---")
for plat in orden:
    horas = sub.loc[sub[COL_PLAT] == plat, "horas_num"]
    print(f"{plat}: media={horas.mean():.1f}h, mediana={horas.median():.1f}h, n={len(horas)}, precio=${PRECIOS_CLP[plat]:,} CLP".replace(",", "."))
