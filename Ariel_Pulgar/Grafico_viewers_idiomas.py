from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Circle


def cargar_datos() -> pd.DataFrame:
	ruta_csv = Path(__file__).with_name(
		"Most watched Twitch streamers, past 365 days - SullyGnome.csv"
	)

	df = pd.read_csv(ruta_csv, header=0)

	if "" in df.columns:
		df = df.drop(columns=[""])

	df.columns = [str(col).strip() for col in df.columns]

	columnas_numericas = [
		"Watch time (mins)",
		"Stream time (mins)",
		"Peak viewers",
		"Average viewers",
		"Followers",
		"Followers gained",
	]

	for columna in columnas_numericas:
		if columna in df.columns:
			df[columna] = pd.to_numeric(df[columna], errors="coerce")

	return df


def obtener_resumen_idiomas(df: pd.DataFrame) -> pd.DataFrame:
	if not {"Language", "Average viewers"}.issubset(df.columns):
		raise ValueError("El CSV debe contener las columnas 'Language' y 'Average viewers'.")

	df = df.copy()
	df["Language"] = df["Language"].fillna("Desconocido")

	resumen_idiomas = (
		df.groupby("Language", dropna=False)["Average viewers"]
		.sum()
		.reset_index(name="Total promedio de espectadores")
		.sort_values("Total promedio de espectadores", ascending=False)
	)
	return resumen_idiomas


def mostrar_resumen(resumen_idiomas: pd.DataFrame) -> None:
	print(resumen_idiomas.to_string(index=False))


def generar_bubble_chart(resumen_idiomas: pd.DataFrame) -> None:
	if resumen_idiomas.empty:
		raise ValueError("No hay datos para generar el bubble chart.")

	valores = resumen_idiomas["Total promedio de espectadores"].to_numpy(dtype=float)
	idiomas = resumen_idiomas["Language"].astype(str).to_list()
	colores = plt.cm.tab20(np.linspace(0, 1, len(idiomas)))

	maximo = float(valores.max()) if len(valores) > 0 else 1.0
	radii = 0.15 + 0.23 * np.sqrt(valores / maximo)
	gap_minimo = 0.02

	indices_ordenados = np.argsort(radii)[::-1]

	def intentar_colocar(radii_actuales: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
		x_pos = np.zeros(len(radii_actuales), dtype=float)
		y_pos = np.zeros(len(radii_actuales), dtype=float)
		colocados: list[int] = []

		if len(indices_ordenados) == 0:
			return x_pos, y_pos

		indice_central = int(indices_ordenados[0])
		x_pos[indice_central] = 0.0
		y_pos[indice_central] = 0.0
		colocados.append(indice_central)

		for indice in indices_ordenados[1:]:
			radio = float(radii_actuales[indice])
			colocado = False

			for radio_espiral in np.arange(0.0, 4.0, 0.01):
				angulos = np.linspace(0, 2 * np.pi, 720, endpoint=False)
				x_candidatos = radio_espiral * np.cos(angulos)
				y_candidatos = radio_espiral * np.sin(angulos)

				for x_candidato, y_candidato in zip(x_candidatos, y_candidatos):
					distancias = np.sqrt(
						(x_pos[colocados] - x_candidato) ** 2 + (y_pos[colocados] - y_candidato) ** 2
					)
					radios_necesarios = radii_actuales[colocados] + radio + gap_minimo

					if np.all(distancias >= radios_necesarios):
						x_pos[indice] = x_candidato
						y_pos[indice] = y_candidato
						colocados.append(indice)
						colocado = True
						break

				if colocado:
					break

			if not colocado:
				return None

		return x_pos, y_pos

	posiciones = None
	radii_trabajo = radii.copy()
	for _ in range(12):
		posiciones = intentar_colocar(radii_trabajo)
		if posiciones is not None:
			break
		radii_trabajo *= 0.93

	if posiciones is None:
		raise RuntimeError("No fue posible ubicar las burbujas sin solapamiento.")

	x, y = posiciones
	radii = radii_trabajo
	extremo = float(np.max(np.sqrt(x**2 + y**2) + radii))
	limite = max(1.2, extremo + 0.08)

	plt.figure(figsize=(14, 8))
	ax = plt.gca()

	for indice, idioma in enumerate(idiomas):
		circulo = Circle(
			(x[indice], y[indice]),
			radius=float(radii[indice]),
			facecolor=colores[indice],
			edgecolor="black",
			linewidth=1.2,
			alpha=0.7,
		)
		ax.add_patch(circulo)

	for indice, idioma in enumerate(idiomas):
		valor_texto = f"{valores[indice]:,.0f}".replace(",", ".")
		etiqueta = f"{idioma}\n{valor_texto}"
		ax.text(
			x[indice],
			y[indice],
			etiqueta,
			ha="center",
			va="center",
			fontsize=8,
			fontweight="bold",
		)

	plt.title("Bubble Chart: Total promedio de espectadores por idioma", fontsize=15)
	ax.set_xticks([])
	ax.set_yticks([])
	ax.set_xlim(-limite, limite)
	ax.set_ylim(-limite, limite)
	ax.set_aspect("equal")
	for spine in ax.spines.values():
		spine.set_visible(False)
	plt.tight_layout()
	plt.show()



if __name__ == "__main__":
	dataframe = cargar_datos()
	resumen = obtener_resumen_idiomas(dataframe)

	generar_bubble_chart(resumen)