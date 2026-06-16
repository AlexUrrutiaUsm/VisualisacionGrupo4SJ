# Reporte de Visualización: Netflix vs. Salario Mínimo (2021)
**Autor: Dante**

Este documento detalla la descripción cualitativa del dataset y el objetivo de la visualización implementada en la carpeta `Dante`.

---

## 1. Descripción Cualitativa del Dataset Seleccionado

La visualización se construye a partir del cruce de dos conjuntos de datos tabulares con referencia temporal al año **2021**:

### A. Dataset de Precios de Suscripción de Netflix
* **Origen:** Recopilación de datos secundarios (fuente original: Comparitech/Kaggle).
* **Tipo de datos:** Tabular, estructurado en formato CSV.
* **Variables utilizadas:**
  * `Country` (Variable categórica nominal / geográfica): Identifica a cada uno de los países.
  * `Cost Per Month - Standard ($)` (Variable numérica continua): El costo mensual de la suscripción estándar expresado en dólares estadounidenses (USD).
* **Calidad y cobertura:** Cuenta con información para **65 países**. Representa una muestra sólida de las principales economías donde opera Netflix de forma regulada. Su principal limitación es la exclusión de gran parte de los países africanos y de Asia Central debido a la falta de presencia del servicio o a la escasez de datos oficiales de precios para ese periodo.

### B. Dataset de Salarios Mínimos (OIT / ILOSTAT)
* **Origen:** Base de datos oficial de la Organización Internacional del Trabajo (OIT), extraída mediante archivo local (`sueldo minimo.csv`).
* **Tipo de datos:** Tabular, estructurado en formato CSV.
* **Variables utilizadas:**
  * `ref_area.label` (Variable categórica nominal / geográfica): Identificador del país.
  * `time` (Variable temporal/ordinal): Año del reporte del salario mínimo. Se filtraron y ordenaron los datos para alinearse de la forma más cercana al año de referencia (2021).
  * `obs_value` (Variable numérica continua): Monto nominal del salario mínimo en moneda local.
  * `note_indicator.label` (Texto / Metadatos): Información clave para la normalización. Especifica la moneda local (ej. *INR*, *BRL*, *EUR*) y la unidad de pago (por hora, diario, mensual).
* **Normalización y Conversión:** 
  Para posibilitar un cruce homogéneo, los salarios se convirtieron a una base **mensualizada** (asumiendo una jornada estándar de 160 horas mensuales para los salarios informados por hora, y 26 días para los diarios) y posteriormente se convirtieron a **USD** utilizando el tipo de cambio oficial promedio de **diciembre de 2021** (fuente: Banco Central Europeo y bancos centrales locales).
* **Limitaciones y Ajuste de Calidad:**
  Economías desarrolladas muy relevantes como **Italia, Suecia, Noruega, Dinamarca, Finlandia, Austria y Singapur** no poseen un salario mínimo nacional regulado por ley (por lo que aparecían vacías en los registros de la OIT). En su lugar, utilizan sistemas de convenios colectivos sectoriales. Para evitar un sesgo de exclusión de estos países en el mapa, se investigaron e inyectaron de forma manual estimaciones promedio del suelo salarial por convenio colectivo para el año 2021.

---

## 2. Objetivo de la Visualización

### ¿Qué pregunta busca responder?
La visualización busca responder a:
> **¿Qué tan accesible o costoso es el servicio de Netflix en términos relativos para un trabajador que percibe el salario mínimo en cada país?**

En lugar de comparar los precios nominales (que harían parecer que el servicio cuesta casi lo mismo en todo el mundo), esta gráfica mide el coste en términos de **esfuerzo financiero local**. Responde específicamente a dos preguntas de métricas cruzadas:
1. *¿Qué porcentaje de un salario mínimo mensual representa el costo del plan estándar de Netflix?*
2. *¿A cuántas horas de trabajo equivale el pago mensual de dicho servicio?*

### ¿A qué público está dirigida?
La visualización está diseñada para:
* **Analistas de mercado y economistas:** Interesados en el estudio del poder adquisitivo global, la economía del comportamiento y la distribución del ingreso.
* **Estudiantes y académicos:** En áreas de visualización de datos, diseño de información y ciencias sociales.
* **Público en general:** Consumidores interesados en comprender la brecha de desigualdad global de una forma cercana, utilizando un servicio de entretenimiento cotidiano como punto de comparación.

### ¿Qué acción o decisión podría apoyar?
* **Estrategias de Precios Corporativos (Pricing):** Provee información clave a los tomadores de decisiones de plataformas digitales globales para definir tarifas dinámicas regionales (precios diferenciados). Permite visualizar dónde el precio del servicio es prohibitivo (ej. Venezuela, donde representa un 730% de un sueldo mínimo) frente a dónde es insignificante (ej. Suiza o Noruega, con menos del 0.5%), lo que ayuda a mitigar la pérdida de suscriptores y a expandir mercados en vías de desarrollo.
* **Análisis de Bienestar y Consumo Familiar:** Apoya a investigadores a ponderar el peso que tienen los gastos hormiga o de entretenimiento digital en el presupuesto de los hogares de distintas regiones geográficas.
* **Políticas de Salario Mínimo y Poder Adquisitivo:** Sirve como herramienta de concientización y debate visual para organizaciones laborales sobre cómo se comporta el poder adquisitivo real frente a bienes globales estandarizados.
