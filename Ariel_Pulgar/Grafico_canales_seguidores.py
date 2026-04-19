from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def cargar_datos() -> pd.DataFrame:
	ruta_csv = Path(__file__).parent / "Most followed streamers on Twitch, past 365 days - SullyGnome.csv"

	df = pd.read_csv(ruta_csv, header=0)

	df.columns = [str(col).strip().strip('"') for col in df.columns]

	columnas_requeridas = ["Channel", "Followers"]
	df = df[columnas_requeridas].copy()
	df["Followers"] = pd.to_numeric(df["Followers"], errors="coerce")
	df["Channel"] = df["Channel"].astype(str).str.strip()
	df = df.dropna(subset=["Channel", "Followers"])
	df = df[df["Channel"] != ""]
	df = df.sort_values("Followers", ascending=False).drop_duplicates(subset=["Channel"])

	return df


def generar_nube_palabras(df: pd.DataFrame, ruta_salida: Path) -> None:
	if df.empty:
		raise ValueError("No hay datos para generar la nube de palabras.")

	canales = df["Channel"].astype(str).tolist()
	followers = df["Followers"].to_numpy(dtype=float)

	max_followers = float(followers.max())
	min_followers = float(followers.min())
	if max_followers == min_followers:
		tamanos = np.full(len(followers), 24.0)
	else:
		tamanos = 12 + (followers - min_followers) * (54 - 12) / (max_followers - min_followers)

	indices = np.argsort(tamanos)[::-1]
	rng = np.random.default_rng(42)
	fig, ax = plt.subplots(figsize=(18, 12))
	ax.set_xlim(-1.05, 1.05)
	ax.set_ylim(-1.05, 1.05)
	ax.set_aspect("equal")
	ax.axis("off")

	colores = plt.cm.turbo(np.linspace(0, 1, len(canales)))
	posiciones_x = []
	posiciones_y = []
	limite = 0.92

	for indice in indices:
		tamano = float(tamanos[indice])
		colocado = False
		for radio_espiral in np.linspace(0.0, limite, 900):
			angulos = np.linspace(0, 2 * np.pi, 180, endpoint=False)
			for angulo in angulos:
				x_candidato = radio_espiral * np.cos(angulo)
				y_candidato = radio_espiral * np.sin(angulo)
				dist = np.sqrt((np.array(posiciones_x) - x_candidato) ** 2 + (np.array(posiciones_y) - y_candidato) ** 2) if posiciones_x else np.array([])
				radio_texto = 0.012 + (tamano / 90.0)
				if dist.size == 0 or np.all(dist >= (radio_texto + 0.012) * 2.0):
					ax.text(
						x_candidato,
						y_candidato,
						canales[indice],
						fontsize=tamano,
						color=colores[indice],
						ha="center",
						va="center",
						fontweight="bold",
						rotation=int(rng.choice([0, 0, 0, -15, 15, -30, 30])),
						alpha=0.95,
					)
					posiciones_x.append(x_candidato)
					posiciones_y.append(y_candidato)
					colocado = True
					break
			if colocado:
				break

		if not colocado:
			ax.text(
				rng.uniform(-0.95, 0.95),
				rng.uniform(-0.95, 0.95),
				canales[indice],
				fontsize=tamano * 0.9,
				color=colores[indice],
				ha="center",
				va="center",
				fontweight="bold",
				alpha=0.85,
			)

	plt.tight_layout()
	plt.show()


if __name__ == "__main__":
	dataframe = cargar_datos()
	ruta_imagen = Path(__file__).with_name("nube_palabras_canales_followers.png")
	generar_nube_palabras(dataframe, ruta_imagen)
