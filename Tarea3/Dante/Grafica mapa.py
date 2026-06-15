"""
Fuentes:
  - Netflix prices: kaggle.com/datasets/prasertk/netflix-subscription-price-in-different-countries
  - Salarios mínimos: ILOSTAT, OIT (descargados manualmente en CSV local para evitar problemas de acceso a la API y garantizar datos históricos de 2021)URL bulk: https://ilostat.ilo.org/data/snapshots/earnings/
"""
import pandas as pd
import plotly.express as px
import sys
import os
import io
import gzip
import urllib.request
import re
import json

# =============================================================================
# CONFIGURACIÓN
# =============================================================================
NETFLIX_CSV_PATH = r"C:\Users\DANTE\Documents\GitHub\VisualisacionGrupo4SJ\Tarea3\Data\netflix price in different countries.csv"
PLAN        = "Standard"          # "Basic", "Standard" o "Premium"
OUTPUT_HTML = "mapa_netflix.html"
OUTPUT_PNG  = "mapa_netflix.png" 

# Ruta del CSV de salarios mínimos local
WAGES_CSV_PATH = r"C:\Users\DANTE\Documents\GitHub\VisualisacionGrupo4SJ\Tarea3\Data\sueldo minimo.csv"

# =============================================================================
# PROCESAMIENTO DE SALARIOS MÍNIMOS LOCALES
# =============================================================================
def load_ilostat_wages(path):
 
    if not os.path.exists(path):
        print(f"\n[ERROR] No se encontró el archivo de salarios mínimos: {path}\n")
        sys.exit(1)

    print(f"  Leyendo CSV de salarios mínimos desde: {path}")
    df_raw = pd.read_csv(path)
    print(f"  Filas en dataset local: {len(df_raw):,}")

    # Mapeo de nombres de países alternativos en el CSV a códigos ISO3
    alt_names = {
        "Bolivia, Plurinational State of": "BOL",
        "Hong Kong, China": "HKG",
        "Republic of Korea": "KOR",
        "Russian Federation": "RUS",
        "Taiwan, China": "TWN",
        "United Kingdom of Great Britain and Northern Ireland": "GBR",
        "United States of America": "USA",
        "Venezuela, Bolivarian Republic of": "VEN",
        "Viet Nam": "VNM",
        "Türkiye": "TUR",
        "Czechia": "CZE",
        "Republic of Moldova": "MDA",
    }
    full_map = {**COUNTRY_TO_ISO3, **alt_names}

    # Asignar código ISO3
    df_raw["iso_alpha3"] = df_raw["ref_area.label"].map(full_map)
    df = df_raw.dropna(subset=["iso_alpha3", "obs_value", "time", "note_indicator.label"]).copy()

    # Filtrar para el año 2021 o anterior (cercano a 2021 para coincidir con los precios de Netflix del 2021)
    df["time"] = pd.to_numeric(df["time"], errors="coerce")
    df = df.dropna(subset=["time"])
    df = df[df["time"] <= 2021]
    df = df.sort_values("time", ascending=False)
    df = df.drop_duplicates(subset="iso_alpha3", keep="first")

    # Extraer la moneda de la columna note_indicator.label
    def extract_currency(note):
        m = re.search(r'\(([A-Z]{3})\)', str(note))
        return m.group(1) if m else None

    df["currency"] = df["note_indicator.label"].apply(extract_currency)
    df = df.dropna(subset=["currency"])
    # Tasas de cambio históricas oficiales de diciembre de 2021 (fuente: Frankfurter/BCE y Bancos Centrales)
    rates = {
        "USD": 1.0,
        "AUD": 1.3787,
        "BGN": 1.7268,
        "BRL": 5.5713,
        "CAD": 1.2708,
        "CHF": 0.91215,
        "CNY": 6.3524,
        "CZK": 21.948,
        "DKK": 6.5658,
        "EUR": 0.88292,
        "GBP": 0.7419,
        "HKD": 7.7991,
        "HRK": 6.6357,
        "HUF": 325.97,
        "IDR": 14215.0,
        "ILS": 3.1043,
        "INR": 74.369,
        "ISK": 130.32,
        "JPY": 115.12,
        "KRW": 1188.75,
        "MXN": 20.434,
        "MYR": 4.166,
        "NOK": 8.8194,
        "NZD": 1.4638,
        "PHP": 51.0,
        "PLN": 4.0587,
        "RON": 4.3696,
        "RUB": 75.313,
        "SEK": 9.0502,
        "SGD": 1.349,
        "THB": 33.245,
        "TRY": 13.45,
        "ZAR": 15.9478,
        "CLP": 849.12,
        "ARS": 101.79,
        "COP": 3970.66,
        "PEN": 3.98,
        "CRC": 633.20,
        "UYU": 43.57,
        "VES": 4650000.0,  
        "EGP": 15.71,
        "PKR": 177.73,
        "SAR": 3.75,
        "UAH": 27.25,
        "VND": 22850.0,
        "NGN": 411.0,
        "KES": 113.0,
        "JOD": 0.709,
        "BOB": 6.91,
        "GTQ": 7.72,
        "HNL": 24.31,
        "PYG": 6850.0,
        "TWD": 27.71,
        "MDL": 17.70,
    }
    print("  Utilizando tasas de cambio históricas de diciembre 2021 para la conversión a USD.")

    # Convertir salario a USD y ajustar por unidad de tiempo si es necesario
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

    df["min_wage"] = df.apply(calculate_usd_monthly_wage, axis=1)
    df = df.dropna(subset=["min_wage"])
    
    # Renombrar columnas para compatibilidad
    df = df.rename(columns={"time": "wage_year"})
    df = df[["iso_alpha3", "min_wage", "wage_year"]].copy()
    df = df[df["min_wage"] > 0]

    # Agregar salarios mínimos estimados (convenios colectivos o aproximados al 2021) (no incluidos en el CSV de OIT)
    estimated_data = [
        {"iso_alpha3": "ITA", "min_wage": 1630.0, "wage_year": 2021},  # Italia (~1440 EUR)
        {"iso_alpha3": "AUT", "min_wage": 2200.0, "wage_year": 2021},  # Austria (~1945 EUR)
        {"iso_alpha3": "SWE", "min_wage": 2040.0, "wage_year": 2021},  # Suecia (~1800 EUR)
        {"iso_alpha3": "NOR", "min_wage": 2800.0, "wage_year": 2021},  # Noruega (~2500 EUR)
        {"iso_alpha3": "FIN", "min_wage": 2150.0, "wage_year": 2021},  # Finlandia (~1900 EUR)
        {"iso_alpha3": "DNK", "min_wage": 2900.0, "wage_year": 2021},  # Dinamarca (~2500 EUR)
        {"iso_alpha3": "SGP", "min_wage": 1200.0, "wage_year": 2021},  # Singapur (~1600 SGD)
        {"iso_alpha3": "GIB", "min_wage": 1750.0, "wage_year": 2021},  # Gibraltar (~£1300 GBP)
        {"iso_alpha3": "LIE", "min_wage": 3900.0, "wage_year": 2021},  # Liechtenstein (~3600 CHF)
    ]
    df_est = pd.DataFrame(estimated_data)
    df = pd.concat([df, df_est], ignore_index=True)

    print(f"  Países con salario mínimo en USD (al 2021, incluyendo estimaciones): {len(df)}")
    print(f"  Años cubiertos: {int(df['wage_year'].min())}–{int(df['wage_year'].max())}")
    return df


# =============================================================================
# CARGA Y LIMPIEZA DE DATOS NETFLIX
# =============================================================================
def load_netflix(path, plan):
    if not os.path.exists(path):
        print(f"\n[ERROR] No se encontró el archivo: {path}")
        print("Descárgalo desde:")
        print("  kaggle.com/datasets/prasertk/netflix-subscription-price-in-different-countries")
        print(f"Y actualiza NETFLIX_CSV_PATH en este script.\n")
        sys.exit(1)

    df = pd.read_csv(path)

    col_map = {
        "Basic":    "Cost Per Month - Basic ($)",
        "Standard": "Cost Per Month - Standard ($)",
        "Premium":  "Cost Per Month - Premium ($)",
    }
    price_col = col_map[plan]

    if price_col not in df.columns:
        possible = [c for c in df.columns if "cost" in c.lower() or "month" in c.lower()]
        print(f"[AVISO] Columna esperada: '{price_col}'")
        print(f"Columnas disponibles: {df.columns.tolist()}")
        print(f"Posibles candidatas:  {possible}")
        sys.exit(1)

    df = df[["Country", price_col]].copy()
    df.columns = ["Country", "precio_usd"]
    df = df.dropna(subset=["precio_usd"])
    df["Country"] = df["Country"].str.strip()
    return df


# =============================================================================
# MAPEO NOMBRE DE PAÍS → ISO3 (para cruzar Netflix con ILOSTAT)
# =============================================================================
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


def merge_and_calculate(df_netflix, df_wages):
    # Agregar ISO3 a Netflix
    df_netflix["iso_alpha3"] = df_netflix["Country"].map(COUNTRY_TO_ISO3)
    sin_iso = df_netflix[df_netflix["iso_alpha3"].isna()]["Country"].tolist()
    if sin_iso:
        print(f"  [AVISO] Países de Netflix sin código ISO (excluidos): {sin_iso}")

    df = pd.merge(
        df_netflix.dropna(subset=["iso_alpha3"]),
        df_wages,
        on="iso_alpha3",
        how="inner"
    )

    df["pct_salario"]   = (df["precio_usd"] / df["min_wage"] * 100).round(2)
    df["horas_trabajo"] = (df["precio_usd"] / (df["min_wage"] / 160)).round(1)
    df["tooltip_precio"]  = df["precio_usd"].apply(lambda x: f"${x:.2f}")
    df["tooltip_salario"] = df["min_wage"].apply(lambda x: f"${x:,.0f}")
    df["tooltip_anio"]    = df["wage_year"].astype(int).astype(str)

    return df.sort_values("pct_salario", ascending=False)


# =============================================================================
# VISUALIZACIÓN
# =============================================================================
def build_map(df, plan):
    import math
    # Crear escala logarítmica para manejar el outlier extremo de Venezuela
    df["pct_salario_log"] = df["pct_salario"].apply(math.log10)

    fig = px.choropleth(
        df,
        locations="iso_alpha3",
        color="pct_salario_log",
        hover_name="Country",
        color_continuous_scale=[
            (0.00, "#f3f0f7"),  # Lavanda grisáceo muy claro (costo relativo bajo)
            (0.20, "#d8daeb"),  # Gris azulado claro
            (0.40, "#b2abd2"),  # Lavanda medio
            (0.60, "#8073ac"),  # Violeta medio
            (0.80, "#542788"),  # Púrpura oscuro
            (1.00, "#2d004d"),  # Púrpura muy profundo (costo relativo alto)
        ],
        range_color=(-0.4, 2.9),  
        custom_data=[
            "tooltip_precio", "tooltip_salario",
            "pct_salario", "horas_trabajo",
            "tooltip_anio"
        ],
        labels={"pct_salario_log": "Costo relativo (escala log)"},
    )

    fig.update_traces(
        hovertemplate=(
            "<b>%{hovertext}</b><br><br>"
            f"Netflix {plan}: <b>%{{customdata[0]}}/mes</b><br>"
            "Salario mínimo: %{customdata[1]}/mes (USD, %{customdata[4]})<br>"
            "Costo Netflix: <b>%{customdata[2]:.2f}% del salario</b><br>"
            "Equivale a: <b>%{customdata[3]:.1f} horas de trabajo</b>"
            "<extra></extra>"
        )
    )

    fig.update_layout(
        title={
            "text": (
                f"¿Cuánto cuesta Netflix respecto al salario mínimo? (2021)<br>"
                f"<sup>Plan {plan} · % del salario mínimo mensual en USD (Año de referencia: 2021) · "
                f"Fuentes: Kaggle/Comparitech (dic. 2021) · ILOSTAT/OIT (salarios e historiales de cambio al 2021)</sup>"
            ),
            "x": 0.5,
            "xanchor": "center",
            "font": {"size": 17},
        },
        geo=dict(
            showframe=False,
            showcoastlines=True,
            coastlinecolor="#cccccc",
            showland=True,
            landcolor="#f5f5f5",
            showocean=True,
            oceancolor="#e8f4f8",
            showlakes=False,
            projection_type="natural earth",
        ),
        coloraxis_colorbar=dict(
            title="% salario<br>mínimo",
            tickvals=[-0.3, 0.0, 0.3, 0.7, 1.0, 1.3, 1.7, 2.0, 2.3, 2.7, 2.9],
            ticktext=["0.5%", "1.0%", "2.0%", "5.0%", "10.0%", "20.0%", "50.0%", "100%", "200%", "500%", "800%"],
            thickness=14,
            len=0.9,
            yanchor="middle",
            y=0.5,
        ),
        margin=dict(t=90, b=20, l=0, r=0),
        height=650,
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(family="Arial, sans-serif", size=13),
    )
    return fig


def print_summary(df, plan):
    print("\n" + "="*70)
    print(f"  RESUMEN — Netflix {plan} vs. Salario Mínimo (ILOSTAT)")
    print("="*70)
    print(f"{'País':<22} {'Netflix':>9} {'Sal. mín.':>11} {'Año':>6} {'% sal.':>8} {'Horas':>7}")
    print("-"*70)
    for _, row in df.iterrows():
        print(
            f"{row['Country']:<22} "
            f"${row['precio_usd']:>7.2f} "
            f"${row['min_wage']:>9,.0f} "
            f"{int(row['wage_year']):>6} "
            f"{row['pct_salario']:>7.2f}% "
            f"{row['horas_trabajo']:>6.1f}h"
        )
    print("-"*70)
    print(f"\n  Más caro (relativo):   {df.iloc[0]['Country']} — {df.iloc[0]['pct_salario']:.2f}%")
    print(f"  Más barato (relativo): {df.iloc[-1]['Country']} — {df.iloc[-1]['pct_salario']:.2f}%")
    print(f"  Países incluidos: {len(df)}")
    print("="*70 + "\n")


# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":
    print(f"\n{'='*55}")
    print(f"  Mapa Netflix vs. Salario Mínimo — Plan {PLAN}")
    print(f"{'='*55}")

    print(f"\n[1/3] Cargando precios Netflix desde: {NETFLIX_CSV_PATH}")
    df_netflix = load_netflix(NETFLIX_CSV_PATH, PLAN)
    print(f"      → {len(df_netflix)} países en el CSV de Netflix")

    print(f"\n[2/3] Cargando salarios mínimos desde CSV local: {WAGES_CSV_PATH}")
    df_wages = load_ilostat_wages(WAGES_CSV_PATH)

    print("\n[3/3] Cruzando datasets y calculando métricas...")
    df = merge_and_calculate(df_netflix, df_wages)
    print(f"      → {len(df)} países con datos cruzados")

    print_summary(df, PLAN)

    fig = build_map(df, PLAN)

    fig.write_html(OUTPUT_HTML)
    print(f"Mapa interactivo guardado: {OUTPUT_HTML}")

    try:
        fig.write_image(OUTPUT_PNG, width=1400, height=700, scale=2)
        print(f"Imagen PNG guardada:       {OUTPUT_PNG}")
    except Exception:
        print("PNG no generado (requerimientos adicionales para exportar imágenes).")

    fig.show()
    print("\nListo. El mapa se abrió en tu navegador.")