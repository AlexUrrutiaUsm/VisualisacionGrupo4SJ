import kagglehub
import pandas as pd
import os
import shutil

# 1. Definir la carpeta donde quieres los datos (tu repo de GitHub)
repo_dir = os.path.dirname(os.path.abspath(__file__))
datos_dir = os.path.join(repo_dir, "data_netflix") # Carpeta organizada

# 2. Descargar al cache de Kaggle (sin el parámetro path)
print("Descargando dataset desde Kaggle...")
cache_path = kagglehub.dataset_download("adnananam/netflix-revenue-and-usage-statistics")

# 3. Mover los archivos a tu carpeta local
if not os.path.exists(datos_dir):
    os.makedirs(datos_dir)

for item in os.listdir(cache_path):
    s = os.path.join(cache_path, item)
    d = os.path.join(datos_dir, item)
    if os.path.isdir(s):
        if os.path.exists(d): shutil.rmtree(d)
        shutil.copytree(s, d)
    else:
        shutil.copy2(s, d)

print(f"Archivos listos en: {datos_dir}")

# 4. Cargar el Excel para empezar a graficar
excel_file = os.path.join(datos_dir, "Netflix Revenue and Usage Statistics.xlsx")
if os.path.exists(excel_file):
    # Usamos ExcelFile para ver todas las pestañas (sheets)
    xl = pd.ExcelFile(excel_file)
    print(f"\nPestañas encontradas: {xl.sheet_names}")
    
    # Por ejemplo, cargamos la de ingresos globales
    # Ajusta 'Netflix Revenue' por el nombre real que veas en sheet_names
    df = xl.parse(xl.sheet_names[0]) 
    
    print("\n--- Vista previa de los datos ---")
    print(df.head())
else:
    print("No se encontró el archivo .xlsx. Revisa el contenido de la carpeta.")