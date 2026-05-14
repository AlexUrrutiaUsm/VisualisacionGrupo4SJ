from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, Normalize, TwoSlopeNorm
from matplotlib.cm import ScalarMappable
import pandas as pd


FONDO_OSCURO = "#1f1f1f"


def cargar_datos() -> pd.DataFrame:
	ruta_csv = "data/Most followed streamers on Twitch, past 365 days - SullyGnome.csv"

	df = pd.read_csv(ruta_csv, header=0)

	df.columns = [str(col).strip().strip('"') for col in df.columns]

	columnas_requeridas = ["Channel", "Followers", "Followers gained"]
	df = df[columnas_requeridas].copy()
	df["Followers"] = pd.to_numeric(df["Followers"], errors="coerce")
	df["Followers gained"] = pd.to_numeric(df["Followers gained"], errors="coerce")
	df["Channel"] = df["Channel"].astype(str).str.strip()
	df = df.dropna(subset=["Channel", "Followers", "Followers gained"])
	df = df[df["Channel"] != ""]
	df = df.sort_values("Followers gained", ascending=False).drop_duplicates(subset=["Channel"])

	return df


def crear_norm(df: pd.DataFrame):
	minimo = float(df["Followers gained"].min())
	maximo = float(df["Followers gained"].max())
	if minimo < 0 < maximo:
		return TwoSlopeNorm(vmin=minimo, vcenter=0, vmax=maximo)
	return Normalize(vmin=minimo, vmax=maximo)


def crear_cmap_centro_estrecho() -> LinearSegmentedColormap:
	return LinearSegmentedColormap.from_list(
		"rojo_amarillo_verde_centro_fino",
		[
			(0.00, "#7f0000"),
			(0.15, "#b2182b"),
			(0.35, "#ef8a62"),
			(0.495, "#fee08b"),
			(0.500, "#ffff00"),
			(0.505, "#d9ef8b"),
			(0.65, "#66bd63"),
			(0.85, "#1a9850"),
			(1.00, "#006837"),
		],
	)


def aplicar_estilo_oscuro(ax: plt.Axes) -> None:
	ax.tick_params(colors="white")
	for spine in ax.spines.values():
		spine.set_color("white")
	ax.xaxis.label.set_color("white")
	ax.yaxis.label.set_color("white")
	ax.title.set_color("white")


def escalar_tamanos_logaritmico(
	valores: pd.Series | np.ndarray,
	tamano_minimo: float,
	tamano_maximo: float,
	percentil_corte: float = 0.95,
) -> np.ndarray:
	valores_array = np.asarray(valores, dtype=float)
	valores_validos = valores_array[np.isfinite(valores_array) & (valores_array > 0)]
	if valores_validos.size == 0:
		return np.full(shape=len(valores_array), fill_value=tamano_minimo)

	techo = float(np.nanpercentile(valores_validos, percentil_corte * 100))
	techo = max(techo, float(np.nanmin(valores_validos)))
	valores_cortados = np.clip(valores_array, 1.0, techo)
	valores_log = np.log1p(valores_cortados)
	min_log = float(np.nanmin(valores_log))
	max_log = float(np.nanmax(valores_log))
	if max_log == min_log:
		return np.full(shape=len(valores_array), fill_value=(tamano_minimo + tamano_maximo) / 2)

	return tamano_minimo + (valores_log - min_log) * (tamano_maximo - tamano_minimo) / (max_log - min_log)


def generar_mapa_calor(df: pd.DataFrame) -> None:
	if df.empty:
		raise ValueError("No hay datos para generar el mapa de calor.")

	data = df.copy().reset_index(drop=True)
	data["label"] = data["Channel"]
	valores = data[["Followers gained"]].to_numpy(dtype=float)
	canales = data["label"].tolist()
	followers = data["Followers"].to_numpy(dtype=float)

	fig, ax = plt.subplots(figsize=(14, max(8, len(canales) * 0.28)))
	fig.patch.set_facecolor(FONDO_OSCURO)
	ax.set_facecolor(FONDO_OSCURO)
	norm = crear_norm(data)
	cmap = crear_cmap_centro_estrecho()
	im = ax.imshow(valores, cmap=cmap, aspect="auto", norm=norm)

	ax.set_yticks(range(len(canales)))
	ax.set_yticklabels(canales)
	ax.set_xticks([0])
	ax.set_xticklabels(["Followers gained"])
	ax.tick_params(axis="y", labelsize=7)
	ax.tick_params(axis="x", labelsize=9)

	min_followers = float(followers.min())
	max_followers = float(followers.max())
	if max_followers == min_followers:
		tamanos_fuente = [9.0] * len(followers)
	else:
		tamanos_fuente = 7 + (followers - min_followers) * (13 - 7) / (max_followers - min_followers)

	for etiqueta, valor_ganado, tamano in zip(ax.get_yticklabels(), data["Followers gained"], tamanos_fuente):
		etiqueta.set_color(cmap(norm(float(valor_ganado))))
		etiqueta.set_fontsize(float(tamano))
		etiqueta.set_fontweight("bold")

	for fila, valor in enumerate(data["Followers gained"]):
		ax.text(
			0,
			fila,
			f"{int(valor):,}".replace(",", "."),
			ha="center",
			va="center",
			color="black",
			fontsize=7,
			fontweight="bold",
		)

	ax.set_title("Mapa de calor: Followers gained por canal", color="white")
	aplicar_estilo_oscuro(ax)
	colorbar = fig.colorbar(im, ax=ax, label="Followers gained")
	colorbar.ax.yaxis.label.set_color("white")
	colorbar.ax.tick_params(colors="white")
	plt.tight_layout()
	ruta_salida = Path(__file__).with_name("mapa_calor_followers_gained.png")
	plt.savefig(ruta_salida, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
	plt.show()


def generar_nube_palabras(df: pd.DataFrame) -> None:
	if df.empty:
		raise ValueError("No hay datos para generar la nube de palabras.")

	data = df.copy().sort_values("Followers", ascending=False).reset_index(drop=True)
	followers = data["Followers"].to_numpy(dtype=float)
	norm = crear_norm(data)
	cmap = crear_cmap_centro_estrecho()

	tamanos_fuente = escalar_tamanos_logaritmico(followers, 6, 18, percentil_corte=0.90)

	fig, ax = plt.subplots(figsize=(20, 14))
	fig.patch.set_facecolor(FONDO_OSCURO)
	ax.set_facecolor(FONDO_OSCURO)
	ax.set_xlim(0, 1)
	ax.set_ylim(0, 1)
	ax.axis("off")

	posiciones = []
	rng = np.random.default_rng(2)
	margen_entre_canales = 0.008

	def calcular_caja_texto(nombre_canal: str, tamano_fuente: float) -> tuple[float, float]:
		longitud = len(nombre_canal)
		ancho = 0.0045 * longitud + (tamano_fuente / 650) + margen_entre_canales
		alto = (tamano_fuente / 190) + 0.010 + margen_entre_canales
		if longitud > 10:
			alto += 0.003
		return ancho, alto

	def hay_colision(x: float, y: float, ancho: float, alto: float) -> bool:
		for px, py, pancho, palto in posiciones:
			if abs(x - px) < (ancho + pancho) * 1.18 and abs(y - py) < (alto + palto) * 1.25:
				return True
		return False

	for indice, fila in data.iterrows():
		canal = str(fila["Channel"])
		valor_ganado = float(fila["Followers gained"])
		tamano = float(tamanos_fuente[indice])
		ancho_texto, alto_texto = calcular_caja_texto(canal, tamano)
		ubicado = False

		for radio in np.linspace(0.03, 0.44, 280):
			for angulo in np.linspace(0, 2 * np.pi, 120, endpoint=False):
				x = 0.5 + radio * np.cos(angulo)
				y = 0.5 + radio * np.sin(angulo)
				if not (0.03 <= x <= 0.97 and 0.03 <= y <= 0.97):
					continue
				if hay_colision(x, y, ancho_texto, alto_texto):
					continue
				ax.text(
					x,
					y,
					canal,
					fontsize=tamano,
					fontweight="bold",
					color=cmap(norm(valor_ganado)),
					ha="center",
					va="center",
					rotation=0,
					transform=ax.transAxes,
				)
				posiciones.append((x, y, ancho_texto, alto_texto))
				ubicado = True
				break
			if ubicado:
				break

		if not ubicado:
			for _ in range(380):
				x = float(rng.uniform(0.05, 0.95))
				y = float(rng.uniform(0.05, 0.95))
				if hay_colision(x, y, ancho_texto, alto_texto):
					continue
				ax.text(
					x,
					y,
					canal,
					fontsize=tamano,
					fontweight="bold",
					color=cmap(norm(valor_ganado)),
					ha="center",
					va="center",
					rotation=0,
					transform=ax.transAxes,
				)
				posiciones.append((x, y, ancho_texto, alto_texto))
				ubicado = True
				break

		if not ubicado:
			tamano_reducido = max(6.5, tamano * 0.7)
			ancho_texto, alto_texto = calcular_caja_texto(canal, tamano_reducido)
			for _ in range(320):
				x = float(rng.uniform(0.04, 0.96))
				y = float(rng.uniform(0.04, 0.96))
				if hay_colision(x, y, ancho_texto, alto_texto):
					continue
				ax.text(
					x,
					y,
					canal,
					fontsize=tamano_reducido,
					fontweight="bold",
					color=cmap(norm(valor_ganado)),
					ha="center",
					va="center",
					rotation=0,
					transform=ax.transAxes,
				)
				posiciones.append((x, y, ancho_texto, alto_texto))
				ubicado = True
				break

		if not ubicado:
			tamano_final = max(5.5, tamano * 0.50)
			ax.text(
				float(rng.uniform(0.05, 0.95)),
				float(rng.uniform(0.05, 0.95)),
				canal,
				fontsize=tamano_final,
				fontweight="bold",
				color=cmap(norm(valor_ganado)),
				ha="center",
				va="center",
				rotation=0,
				transform=ax.transAxes,
			)

	fig.suptitle("Nube de palabras: tamaño = Followers, color = Followers gained", fontsize=14, color="white")
	sm = ScalarMappable(norm=norm, cmap=cmap)
	sm.set_array([])
	colorbar = fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.02, label="Followers gained")
	colorbar.ax.yaxis.label.set_color("white")
	colorbar.ax.tick_params(colors="white")
	aplicar_estilo_oscuro(ax)
	ruta_salida = Path(__file__).with_name("nube_palabras_followers.png")
	plt.tight_layout()
	plt.savefig(ruta_salida, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
	plt.show()


def generar_bubble_chart(df: pd.DataFrame) -> None:
	if df.empty:
		raise ValueError("No hay datos para generar el bubble chart.")

	data = df.copy().reset_index(drop=True)
	norm = crear_norm(data)
	cmap = crear_cmap_centro_estrecho()

	followers = data["Followers"].to_numpy(dtype=float)
	tamanos_burbuja = escalar_tamanos_logaritmico(followers, 10, 100)

	fig, ax = plt.subplots(figsize=(100, 100))
	fig.patch.set_facecolor(FONDO_OSCURO)
	ax.set_facecolor(FONDO_OSCURO)
	scatter = ax.scatter(
		data["Followers gained"],
		data["Followers"],
		s=tamanos_burbuja,
		c=data["Followers gained"],
		cmap=cmap,
		norm=norm,
		alpha=1,
		edgecolors="black",
		linewidths=0.35,
	)

	for _, fila in data.iterrows():
		ax.text(
			float(fila["Followers gained"]),
			float(fila["Followers"]),
			str(fila["Channel"]),
			fontsize=6,
			ha="center",
			va="center",
			color="white",
		)

	ax.set_title("Bubble chart: Followers gained vs Followers", color="white")
	ax.set_xlabel("Followers gained", color="white")
	ax.set_ylabel("Followers", color="white")
	ax.grid(alpha=0.2)
	aplicar_estilo_oscuro(ax)
	colorbar = fig.colorbar(scatter, ax=ax, label="Followers gained")
	colorbar.ax.yaxis.label.set_color("white")
	colorbar.ax.tick_params(colors="white")

	ruta_salida = Path(__file__).with_name("bubble_followers.png")
	plt.tight_layout()
	plt.savefig(ruta_salida, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
	plt.show()


def generar_visualizacion(df: pd.DataFrame, tipo: str = "nube") -> None:
	tipo_normalizado = tipo.strip().lower()
	if tipo_normalizado == "mapa":
		generar_mapa_calor(df)
		return
	if tipo_normalizado in {"bubble", "burble", "burbuja"}:
		generar_bubble_chart(df)
		return
	if tipo_normalizado in {"nube", "wordcloud"}:
		generar_nube_palabras(df)
		return
	raise ValueError("Tipo no válido. Usa: 'nube', 'bubble' o 'mapa'.")


if __name__ == "__main__":
	dataframe = cargar_datos()
	tipo_visualizacion = "nube"
	generar_visualizacion(dataframe, tipo=tipo_visualizacion)