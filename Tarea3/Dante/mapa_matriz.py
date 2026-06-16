"""
Visualización de Mapa de Matriz de Puntos (Dot Matrix Map):
Costo relativo de Netflix vs. Salario Mínimo en USD (2021).
El mapa terrestre se representa como una rejilla regular de puntos,
coloreados en escala azul/verde según el costo relativo, y gris claro
para los territorios sin datos (como océanos no graficados y países sin registrar).
"""
import pandas as pd
import plotly.graph_objects as go
import sys
import os
import re
import math
import json
import urllib.request

# =============================================================================
# CONFIGURACIÓN
# =============================================================================
NETFLIX_CSV_PATH = r"C:\Users\DANTE\Documents\GitHub\VisualisacionGrupo4SJ\Tarea3\Data\netflix price in different countries.csv"
WAGES_CSV_PATH = r"C:\Users\DANTE\Documents\GitHub\VisualisacionGrupo4SJ\Tarea3\Data\sueldo minimo.csv"
GEOJSON_CACHE_PATH = r"C:\Users\DANTE\Documents\GitHub\VisualisacionGrupo4SJ\Tarea3\Data\world-countries.json"
PLAN = "Standard"
OUTPUT_HTML = "mapa_matriz.html"
OUTPUT_PNG  = "mapa_matriz.png"

COUNTRY_TO_ISO3 = {
    "Argentina": "ARG", "Australia": "AUS", "Austria": "AUT",
    "Belgium": "BEL", "Bolivia": "BOL", "Brazil": "BRA",
    "Bulgaria": "BGR", "Canada": "CAN", "Chile": "CHL",
    "Colombia": "COL", "Costa Rica": "CRI", "Croatia": "HRV", "Denmark": "DNK",
    "Czech Republic": "CZE", "Czechia": "CZE", "Ecuador": "ECU", "Egypt": "EGY",
    "Estonia": "EST", "Finland": "FIN", "France": "FRA",
    "Germany": "DEU", "Gibraltar": "GIB", "Greece": "GRC", "Guatemala": "GTM", "Honduras": "HND", "Hong Kong": "HKG",
    "Hungary": "HUN", "Iceland": "ISL", "India": "IND", "Indonesia": "IDN",
    "Ireland": "IRL", "Israel": "ISR", "Italy": "ITA",
    "Japan": "JPN", "Jordan": "JOR", "Kenya": "KEN",
    "Latvia": "LVA", "Liechtenstein": "LIE", "Lithuania": "LTU", "Luxembourg": "LUX",
    "Malaysia": "MYS", "Mexico": "MEX", "Moldova": "MDA", "Monaco": "MCO", "Netherlands": "NLD",
    "New Zealand": "NZL", "Nigeria": "NGA", "Norway": "NOR",
    "Pakistan": "PAK", "Paraguay": "PRY", "Peru": "PER", "Philippines": "PHL",
    "Poland": "POL", "Portugal": "PRT", "Romania": "ROU",
    "Russia": "RUS", "San Marino": "SMR", "Saudi Arabia": "SAU", "Serbia": "SRB",
    "Singapore": "SGP", "Slovakia": "SVK", "Slovenia": "SVN",
    "South Africa": "ZAF", "South Korea": "KOR", "Spain": "ESP",
    "Sweden": "SWE", "Switzerland": "CHE", "Taiwan": "TWN",
    "Thailand": "THA", "Turkey": "TUR", "Ukraine": "UKR",
    "United Kingdom": "GBR", "United States": "USA", "Uruguay": "URY",
    "Venezuela": "VEN", "Vietnam": "VNM",
}

def load_data():
    # 1. Netflix Prices
    df_netflix = pd.read_csv(NETFLIX_CSV_PATH)
    col_map = {"Basic": "Cost Per Month - Basic ($)", "Standard": "Cost Per Month - Standard ($)", "Premium": "Cost Per Month - Premium ($)"}
    df_netflix = df_netflix[["Country", col_map[PLAN]]].copy()
    df_netflix.columns = ["Country", "precio_usd"]
    df_netflix = df_netflix.dropna()
    df_netflix["Country"] = df_netflix["Country"].str.strip()
    df_netflix["iso_alpha3"] = df_netflix["Country"].map(COUNTRY_TO_ISO3)
    df_netflix = df_netflix.dropna(subset=["iso_alpha3"])

    # 2. Minimum Wages
    df_wages_raw = pd.read_csv(WAGES_CSV_PATH)
    alt_names = {
        "Bolivia, Plurinational State of": "BOL", "Hong Kong, China": "HKG", "Republic of Korea": "KOR",
        "Russian Federation": "RUS", "Taiwan, China": "TWN", "United Kingdom of Great Britain and Northern Ireland": "GBR",
        "United States of America": "USA", "Venezuela, Bolivarian Republic of": "VEN", "Viet Nam": "VNM",
        "Türkiye": "TUR", "Czechia": "CZE", "Republic of Moldova": "MDA"
    }
    full_map = {**COUNTRY_TO_ISO3, **alt_names}
    df_wages_raw["iso_alpha3"] = df_wages_raw["ref_area.label"].map(full_map)
    df_w = df_wages_raw.dropna(subset=["iso_alpha3", "obs_value", "time", "note_indicator.label"]).copy()
    df_w["time"] = pd.to_numeric(df_w["time"], errors="coerce")
    df_w = df_w.dropna(subset=["time"])
    df_w = df_w[df_w["time"] <= 2021]
    df_w = df_w.sort_values("time", ascending=False).drop_duplicates(subset="iso_alpha3", keep="first")

    def extract_currency(note):
        m = re.search(r'\(([A-Z]{3})\)', str(note))
        return m.group(1) if m else None

    df_w["currency"] = df_w["note_indicator.label"].apply(extract_currency)
    df_w = df_w.dropna(subset=["currency"])

    rates = {
        "USD": 1.0, "AUD": 1.3787, "BGN": 1.7268, "BRL": 5.5713, "CAD": 1.2708, "CHF": 0.91215,
        "CNY": 6.3524, "CZK": 21.948, "DKK": 6.5658, "EUR": 0.88292, "GBP": 0.7419, "HKD": 7.7991,
        "HRK": 6.6357, "HUF": 325.97, "IDR": 14215.0, "ILS": 3.1043, "INR": 74.369, "ISK": 130.32,
        "JPY": 115.12, "KRW": 1188.75, "MXN": 20.434, "MYR": 4.166, "NOK": 8.8194, "NZD": 1.4638,
        "PHP": 51.0, "PLN": 4.0587, "RON": 4.3696, "RUB": 75.313, "SEK": 9.0502, "SGD": 1.349,
        "THB": 33.245, "TRY": 13.45, "ZAR": 15.9478, "CLP": 849.12, "ARS": 101.79, "COP": 3970.66,
        "PEN": 3.98, "CRC": 633.20, "UYU": 43.57, "VES": 4650000.0, "EGP": 15.71, "PKR": 177.73,
        "SAR": 3.75, "UAH": 27.25, "VND": 22850.0, "NGN": 411.0, "KES": 113.0, "JOD": 0.709,
        "BOB": 6.91, "GTQ": 7.72, "HNL": 24.31, "PYG": 6850.0, "TWD": 27.71, "MDL": 17.70
    }

    def calculate_usd_monthly_wage(row):
        rate = rates.get(row["currency"])
        if not rate:
            return None
        val = float(row["obs_value"])
        note = str(row["note_indicator.label"]).lower()
        if "per hour" in note:
            val = val * 40 * 4.33
        elif "per day" in note:
            val = val * 6 * 4.33
        return val / rate

    df_w["min_wage"] = df_w.apply(calculate_usd_monthly_wage, axis=1)
    df_w = df_w.dropna(subset=["min_wage"]).rename(columns={"time": "wage_year"})
    df_w = df_w[["iso_alpha3", "min_wage", "wage_year"]].copy()

    # Manual Estimates
    estimated_data = [
        {"iso_alpha3": "ITA", "min_wage": 1630.0, "wage_year": 2021},
        {"iso_alpha3": "AUT", "min_wage": 2200.0, "wage_year": 2021},
        {"iso_alpha3": "SWE", "min_wage": 2040.0, "wage_year": 2021},
        {"iso_alpha3": "NOR", "min_wage": 2800.0, "wage_year": 2021},
        {"iso_alpha3": "FIN", "min_wage": 2150.0, "wage_year": 2021},
        {"iso_alpha3": "DNK", "min_wage": 2900.0, "wage_year": 2021},
        {"iso_alpha3": "SGP", "min_wage": 1200.0, "wage_year": 2021},
        {"iso_alpha3": "GIB", "min_wage": 1750.0, "wage_year": 2021},
        {"iso_alpha3": "LIE", "min_wage": 3900.0, "wage_year": 2021}
    ]
    df_est = pd.DataFrame(estimated_data)
    df_w = pd.concat([df_w, df_est], ignore_index=True)

    # Merge
    df = pd.merge(df_netflix, df_w, on="iso_alpha3", how="inner")
    df["pct_salario"] = (df["precio_usd"] / df["min_wage"] * 100).round(2)
    df["horas_trabajo"] = (df["precio_usd"] / (df["min_wage"] / 160)).round(1)
    df["tooltip_precio"]  = df["precio_usd"].apply(lambda x: f"${x:.2f}")
    df["tooltip_salario"] = df["min_wage"].apply(lambda x: f"${x:,.0f}")
    df["tooltip_anio"]    = df["wage_year"].astype(int).astype(str)
    
    return df

def load_geojson():
    if os.path.exists(GEOJSON_CACHE_PATH):
        print(f"  Cargando GeoJSON local desde: {GEOJSON_CACHE_PATH}")
        with open(GEOJSON_CACHE_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    else:
        URL = "https://raw.githubusercontent.com/python-visualization/folium/master/examples/data/world-countries.json"
        print(f"  Descargando GeoJSON desde: {URL}")
        with urllib.request.urlopen(URL) as response:
            data = json.loads(response.read().decode('utf-8'))
        os.makedirs(os.path.dirname(GEOJSON_CACHE_PATH), exist_ok=True)
        with open(GEOJSON_CACHE_PATH, "w", encoding="utf-8") as f:
            json.dump(data, f)
        print(f"  GeoJSON guardado localmente en: {GEOJSON_CACHE_PATH}")
        return data

def generate_grid_mapping():
    geojson_data = load_geojson()
    features = geojson_data["features"]

    print("  Precomputando bounding boxes de los países...")
    precomputed_features = []
    for feature in features:
        geom = feature["geometry"]
        g_type = geom["type"]
        coords = geom["coordinates"]
        polys = []
        if g_type == "Polygon":
            ext_ring = coords[0]
            lons = [p[0] for p in ext_ring]
            lats = [p[1] for p in ext_ring]
            bbox = (min(lons), min(lats), max(lons), max(lats))
            polys.append((bbox, ext_ring))
        elif g_type == "MultiPolygon":
            for poly in coords:
                ext_ring = poly[0]
                lons = [p[0] for p in ext_ring]
                lats = [p[1] for p in ext_ring]
                bbox = (min(lons), min(lats), max(lons), max(lats))
                polys.append((bbox, ext_ring))
        precomputed_features.append({
            "id": feature["id"],
            "name": feature["properties"]["name"],
            "polys": polys
        })

    def point_in_polygon(x, y, poly):
        n = len(poly)
        inside = False
        p1x, p1y = poly[0]
        for i in range(n + 1):
            p2x, p2y = poly[i % n]
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != y and p2y != p1y:
                            xints = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        else:
                            xints = p1x
                        if p1x == p2x or x <= xints:
                            inside = not inside
            p1x, p1y = p2x, p2y
        return inside

    # Generar rejilla regular de puntos
    # Resolución: 2.0 grados (crea una malla tupida y estética)
    print("  Generando rejilla regular de puntos (matriz)...")
    grid_points = []
    for lat in range(-56, 81, 2):
        for lon in range(-178, 180, 2):
            grid_points.append((lon, lat))

    print(f"  Verificando {len(grid_points)} puntos contra los límites terrestres...")
    point_countries = []
    for lon, lat in grid_points:
        found = False
        for pf in precomputed_features:
            for bbox, ring in pf["polys"]:
                if bbox[0] <= lon <= bbox[2] and bbox[1] <= lat <= bbox[3]:
                    if point_in_polygon(lon, lat, ring):
                        point_countries.append((lon, lat, pf["id"], pf["name"]))
                        found = True
                        break
            if found:
                break
            
    print(f"  Rejilla generada. Puntos en territorio firme: {len(point_countries)}")
    return pd.DataFrame(point_countries, columns=["lon", "lat", "iso_alpha3", "Country"])

def build_matrix_map(df_data, df_grid):
    df_merged = pd.merge(df_grid, df_data, on="iso_alpha3", how="left", suffixes=("_grid", ""))
    
    # Separar puntos en con datos y sin datos (fondo gris)
    df_no_data = df_merged[df_merged["pct_salario"].isna()].copy()
    df_with_data = df_merged[df_merged["pct_salario"].notna()].copy()
    df_with_data["pct_salario_log"] = df_with_data["pct_salario"].apply(math.log10)

    # Texto tooltip personalizado
    df_with_data["hover_text"] = (
        "<b>" + df_with_data["Country"] + "</b><br><br>"
        "Netflix " + PLAN + ": <b>" + df_with_data["tooltip_precio"] + "/mes</b><br>"
        "Salario mínimo: " + df_with_data["tooltip_salario"] + "/mes (USD, " + df_with_data["tooltip_anio"] + ")<br>"
        "Costo Netflix: <b>" + df_with_data["pct_salario"].astype(str) + "% del salario</b><br>"
        "Equivale a: <b>" + df_with_data["horas_trabajo"].astype(str) + " horas de trabajo</b>"
    )

    fig = go.Figure()

    # 1. Capa de fondo terrestre (puntos grises claros)
    fig.add_trace(go.Scattergeo(
        lon=df_no_data["lon"],
        lat=df_no_data["lat"],
        mode="markers",
        marker=dict(
            size=3.5,
            color="#e2e4e6",  # Gris plata suave
            symbol="circle",
            line=dict(width=0)
        ),
        hoverinfo="skip",
        showlegend=False
    ))

    # 2. Capa de países con datos (escala azul/verde degradada)
    fig.add_trace(go.Scattergeo(
        lon=df_with_data["lon"],
        lat=df_with_data["lat"],
        mode="markers",
        marker=dict(
            size=5.5,
            color=df_with_data["pct_salario_log"],
            colorscale=[
                (0.00, "#a1dab4"),  # Verde azulado muy claro (costo bajo)
                (0.25, "#41b6c4"),  # Turquesa claro
                (0.50, "#1d91c0"),  # Azul medio
                (0.75, "#225ea8"),  # Azul real
                (1.00, "#081d58"),  # Azul marino muy oscuro (costo alto/extremo)
            ],
            cmin=-0.4,
            cmax=2.9,
            showscale=True,
            colorbar=dict(
                title="% salario<br>mínimo",
                tickvals=[-0.3, 0.0, 0.3, 0.7, 1.0, 1.3, 1.7, 2.0, 2.3, 2.7, 2.9],
                ticktext=["0.5%", "1.0%", "2.0%", "5.0%", "10.0%", "20.0%", "50.0%", "100%", "200%", "500%", "800%"],
                thickness=14,
                len=0.9,
                yanchor="middle",
                y=0.5,
            ),
            line=dict(width=0)
        ),
        text=df_with_data["hover_text"],
        hoverinfo="text",
        showlegend=False
    ))

    fig.update_layout(
        title={
            "text": (
                f"Matriz de Puntos: ¿Cuánto cuesta Netflix respecto al salario mínimo? (2021)<br>"
                f"<sup>Plan {PLAN} · % del salario mínimo mensual en USD (Territorios representados en malla de círculos)</sup>"
            ),
            "x": 0.5,
            "xanchor": "center",
            "font": {"size": 16},
        },
        geo=dict(
            showframe=False,
            showcoastlines=False,  # Ocultar costas para destacar la matriz de puntos
            showland=False,       # Ocultar relleno terrestre por defecto
            showocean=False,      # Mantener el océano transparente/blanco
            projection_type="natural earth",
        ),
        margin=dict(t=95, b=20, l=10, r=10),
        height=650,
        paper_bgcolor="white",
        font=dict(family="Arial, sans-serif", size=13),
    )
    return fig

if __name__ == "__main__":
    print("Cargando salarios mínimos y suscripciones de Netflix...")
    df_data = load_data()
    print("Mapeando límites mundiales a matriz de puntos...")
    df_grid = generate_grid_mapping()
    
    print("Construyendo Mapa de Matriz de Puntos...")
    fig = build_matrix_map(df_data, df_grid)
    
    # Guardar salidas
    fig.write_html(OUTPUT_HTML)
    print(f"Mapa matriz interactivo guardado como: {OUTPUT_HTML}")
    
    try:
        fig.write_image(OUTPUT_PNG, width=1400, height=700, scale=2)
        print(f"Mapa matriz en imagen estática guardado como: {OUTPUT_PNG}")
    except Exception as e:
        print("Imagen PNG estática no generada (requiere kaleido: pip install kaleido)")
        
    fig.show()
    print("Listo. Se abrió el mapa en tu navegador.")
