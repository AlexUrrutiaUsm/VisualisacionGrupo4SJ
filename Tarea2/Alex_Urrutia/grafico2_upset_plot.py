import os
from collections import Counter
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

DATA_PATH = os.path.join(
    os.path.dirname(__file__), '..', 'data',
    'Encuesta sobre Hábitos de Consumo de Streaming y Visualización de Datos.csv'
)

df = pd.read_csv(DATA_PATH)

FACTORES_COL = df.columns[5]

df_clean = df[df[FACTORES_COL].notna() & (df[FACTORES_COL].str.strip() != '')].copy()

ABBREV = {
    'Género/Categoría':                                 'Género/Categ.',
    'Recomendaciones personalizadas de la plataforma':  'Recomendaciones',
    'Tendencias actuales/Popularidad':                  'Tendencias',
    'Comentarios o reseñas de amigos/redes sociales':   'Comentarios/Reseñas',
    'Calificación/Puntuación de otros usuarios':        'Calificación',
    'Duración del contenido':                           'Duración',
    'Disponibilidad en 4K/HDR/Audio espacial':          'Calidad 4K/HDR',
}

def abreviar(lista):
    return frozenset(ABBREV.get(f.strip(), f.strip()) for f in lista)

memberships = df_clean[FACTORES_COL].str.split(';').apply(abreviar)

combo_counts = Counter(memberships)
combos_sorted = sorted(combo_counts.items(), key=lambda x: -x[1])

all_cats = sorted({cat for combo in combo_counts for cat in combo})
n_cats   = len(all_cats)
n_combos = len(combos_sorted)
cat_idx  = {cat: i for i, cat in enumerate(all_cats)}

set_counts = {
    cat: sum(count for combo, count in combo_counts.items() if cat in combo)
    for cat in all_cats
}

COLOR = '#4e79a7'
GRAY  = '#cccccc'

fig = plt.figure(figsize=(max(14, n_combos * 0.8), 8))
fig.patch.set_facecolor('white')

gs = GridSpec(
    2, 2, figure=fig,
    height_ratios=[2.5, 1.5],
    width_ratios=[1.8, 4],
    hspace=0.06, wspace=0.06,
)

ax_bars    = fig.add_subplot(gs[0, 1])
ax_matrix  = fig.add_subplot(gs[1, 1])
ax_setsize = fig.add_subplot(gs[1, 0])
ax_empty   = fig.add_subplot(gs[0, 0])
ax_empty.axis('off')

counts = [c for _, c in combos_sorted]
ax_bars.bar(range(n_combos), counts, color=COLOR, width=0.6)
ax_bars.set_xlim(-0.5, n_combos - 0.5)
ax_bars.set_xticks([])
ax_bars.set_ylabel('Tamaño de intersección', fontsize=10)
ax_bars.spines['top'].set_visible(False)
ax_bars.spines['right'].set_visible(False)
ax_bars.set_facecolor('white')
for i, v in enumerate(counts):
    ax_bars.text(i, v + 0.05, str(v), ha='center', va='bottom', fontsize=8)

ax_matrix.set_facecolor('white')
ax_matrix.set_xlim(-0.5, n_combos - 0.5)
ax_matrix.set_ylim(-0.5, n_cats - 0.5)
ax_matrix.set_xticks([])
ax_matrix.set_yticks(range(n_cats))
ax_matrix.set_yticklabels(all_cats, fontsize=9)
ax_matrix.spines['top'].set_visible(False)
ax_matrix.spines['right'].set_visible(False)
ax_matrix.spines['bottom'].set_visible(False)

for i in range(n_cats):
    if i % 2 == 0:
        ax_matrix.axhspan(i - 0.5, i + 0.5, color='#f5f5f5', zorder=0)

for col_idx, (combo, _) in enumerate(combos_sorted):
    present = sorted(cat_idx[c] for c in combo)
    absent  = [cat_idx[c] for c in all_cats if c not in combo]
    for row in absent:
        ax_matrix.plot(col_idx, row, 'o', color=GRAY, markersize=8, zorder=2)
    if len(present) > 1:
        ax_matrix.plot(
            [col_idx, col_idx], [min(present), max(present)],
            color=COLOR, linewidth=3, zorder=3,
        )
    for row in present:
        ax_matrix.plot(col_idx, row, 'o', color=COLOR, markersize=10, zorder=4)

ax_setsize.set_facecolor('white')
ax_setsize.barh(
    range(n_cats),
    [set_counts[cat] for cat in all_cats],
    color=COLOR, height=0.5,
)
ax_setsize.set_ylim(-0.5, n_cats - 0.5)
ax_setsize.set_yticks([])
ax_setsize.set_xlabel('Tamaño del conjunto', fontsize=10)
ax_setsize.invert_xaxis()
ax_setsize.spines['top'].set_visible(False)
ax_setsize.spines['left'].set_visible(False)

plt.suptitle(
    'Combinaciones de Factores que Influyen en la Elección de Contenido',
    fontsize=13, fontweight='bold',
)

OUTPUT_PATH = os.path.join(os.path.dirname(__file__), 'grafico2_upset_plot.png')
plt.savefig(OUTPUT_PATH, bbox_inches='tight', dpi=150)
plt.show()
print(f"Gráfico guardado en: {OUTPUT_PATH}")
