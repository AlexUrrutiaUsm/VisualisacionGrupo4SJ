"""
Gráfico 3 – Diagrama Sankey: Horas de consumo semanal → Factores de decisión.

Cruza las dos preguntas:
  - "En promedio, ¿cuántas horas dedicas a consumir contenido … a la semana?"
  - "Al momento de elegir qué ver … ¿cuál de los factores influye más …?"

El grosor de cada flujo representa cuántas personas combinan ese rango de horas
con ese factor de decisión, permitiendo ver si los usuarios de alto consumo se
guían más por recomendaciones/tendencias que los de bajo consumo.

El gráfico se guarda como HTML interactivo (abrir en cualquier navegador).

Requiere: plotly  → pip install plotly
"""

import os
import pandas as pd
import plotly.graph_objects as go

# ── Datos ─────────────────────────────────────────────────────────────────────
DATA_PATH = os.path.join(
    os.path.dirname(__file__), '..', 'data',
    'Encuesta sobre Hábitos de Consumo de Streaming y Visualización de Datos.csv'
)

df = pd.read_csv(DATA_PATH)

HORAS_COL    = df.columns[4]   # horas de consumo semanal
FACTORES_COL = df.columns[5]   # factores de decisión (hasta 3, separados por ";")

# Descartar filas sin datos en alguna de las dos columnas
df_clean = df[
    df[HORAS_COL].notna()    & (df[HORAS_COL].str.strip()    != '') &
    df[FACTORES_COL].notna() & (df[FACTORES_COL].str.strip() != '')
].copy()

# ── Abreviaturas ─────────────────────────────────────────────────────────────
ABBREV = {
    'Género/Categoría':                                 'Género/Categ.',
    'Recomendaciones personalizadas de la plataforma':  'Recomendaciones',
    'Tendencias actuales/Popularidad':                  'Tendencias',
    'Comentarios o reseñas de amigos/redes sociales':   'Comentarios/Reseñas',
    'Calificación/Puntuación de otros usuarios':        'Calificación',
    'Duración del contenido':                           'Duración',
    'Disponibilidad en 4K/HDR/Audio espacial':          'Calidad 4K/HDR',
}

# ── Expandir filas (una por cada factor elegido por persona) ─────────────────
rows = []
for _, row in df_clean.iterrows():
    horas = row[HORAS_COL].strip()
    for factor in row[FACTORES_COL].split(';'):
        factor_clean = ABBREV.get(factor.strip(), factor.strip())
        rows.append({'horas': horas, 'factor': factor_clean})

expanded = pd.DataFrame(rows)
flow = expanded.groupby(['horas', 'factor']).size().reset_index(name='count')

# ── Ordenar nodos ─────────────────────────────────────────────────────────────
HORAS_ORDER = [
    'Menos de 3 horas',
    '3 a 7 horas',
    '8 a 15 horas',
    '16 a 25 horas',
    'Más de 25 horas',
]
horas_present  = [h for h in HORAS_ORDER if h in expanded['horas'].unique()]
factores_order = sorted(expanded['factor'].unique())

all_nodes = horas_present + factores_order
node_idx  = {name: i for i, name in enumerate(all_nodes)}

# ── Paleta de colores ─────────────────────────────────────────────────────────
HORAS_COLORS = ['#4e79a7', '#f28e2b', '#e15759', '#76b7b2', '#59a14f']
FACTOR_COLORS = [
    '#edc948', '#b07aa1', '#ff9da7', '#9c755f',
    '#bab0ac', '#d37295', '#a0cbe8',
]

node_colors = (
    HORAS_COLORS[:len(horas_present)] +
    FACTOR_COLORS[:len(factores_order)]
)

def hex_to_rgba(hex_color: str, alpha: float = 0.45) -> str:
    h = hex_color.lstrip('#')
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f'rgba({r},{g},{b},{alpha})'

# ── Construir enlaces ─────────────────────────────────────────────────────────
sources, targets, values, link_colors = [], [], [], []

for _, row in flow.iterrows():
    horas_name  = row['horas']
    factor_name = row['factor']

    if horas_name not in node_idx or factor_name not in node_idx:
        continue

    sources.append(node_idx[horas_name])
    targets.append(node_idx[factor_name])
    values.append(int(row['count']))

    color_hex = HORAS_COLORS[horas_present.index(horas_name)]
    link_colors.append(hex_to_rgba(color_hex))

# ── Figura ────────────────────────────────────────────────────────────────────
fig = go.Figure(
    go.Sankey(
        arrangement='snap',
        node=dict(
            pad=18,
            thickness=22,
            line=dict(color='black', width=0.5),
            label=all_nodes,
            color=node_colors,
        ),
        link=dict(
            source=sources,
            target=targets,
            value=values,
            color=link_colors,
        ),
    )
)

fig.update_layout(
    title=dict(
        text='<b>Horas de Consumo Semanal → Factores de Decisión</b>',
        font=dict(size=17),
    ),
    font=dict(size=12, family='Arial'),
    height=620,
    margin=dict(l=20, r=20, t=60, b=20),
)

# ── Guardar y mostrar ─────────────────────────────────────────────────────────
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), 'grafico3_sankey.html')
fig.write_html(OUTPUT_PATH)
fig.show()
print(f"Gráfico guardado en: {OUTPUT_PATH}")
