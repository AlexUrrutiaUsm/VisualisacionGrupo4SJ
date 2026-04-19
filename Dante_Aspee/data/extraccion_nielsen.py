import requests
from bs4 import BeautifulSoup
import pandas as pd
import time

def extract_nielsen_data(url):
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"}
    try:
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            table = soup.find('table')
            if table:
                df = pd.read_html(str(table))[0]
                return df
    except Exception as e:
        print(f"Error en {url}: {e}")
    return None

# --- CONFIGURACIÓN DEL CRAWLER ---
años = [2022, 2023, 2024, 2025]
meses = ["january", "february", "march", "april", "may", "june", 
         "july", "august", "september", "october", "november", "december"]

todas_las_tablas = []

print("🚀 Iniciando extracción masiva...")

for año in años:
    for mes in meses:
        # Nielsen suele usar este formato de URL
        url = f"https://www.nielsen.com/insights/{año}/the-gauge-{mes}-{año}/"
        
        print(f"Buscando: {mes} {año}...", end=" ")
        df_mes = extract_nielsen_data(url)
        
        if df_mes is not None:
            # Añadimos columnas de fecha para no perder el contexto al unir
            df_mes['Month'] = mes
            df_mes['Year'] = año
            todas_las_tablas.append(df_mes)
            print("✅ Encontrado")
        else:
            print("❌ No disponible")
        
        # Pausa de seguridad para que no nos bloqueen la IP (importante como programador)
        time.sleep(1)

# --- UNIR Y GUARDAR ---
if todas_las_tablas:
    df_final = pd.concat(todas_las_tablas, ignore_index=True)
    df_final.to_csv("nielsen_history_2022_2025.csv", index=False)
    print("\n✅ ¡Listo! Data guardada en 'nielsen_history_2022_2025.csv'")
    print(df_final.head())
else:
    print("\n⚠️ No se pudo extraer ninguna tabla. Verifica las URLs en nielsen.com")