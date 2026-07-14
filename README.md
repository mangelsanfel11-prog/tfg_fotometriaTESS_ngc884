# Fotometría de TESS y análisis de frecuencias en el cúmulo abierto NGC 884

Este repositorio contiene un script de Python desarrollado para mi Trabajo 
de Fin de Grado (TFG) en el Grado en Física (Julio 2026, Universitat de València)

## Contenido del repositorio

*  **`ClearcurvesSG_FrequencyAnalysis_sector.py`**: Pipeline automatizado no interactivo que procesa en lote todas las curvas FITS depositadas en la carpeta 'Datos_sector' y almacena los resultados en carpetas estructuradas en la carpeta 'Resultados_sector'.

El flujo de procesamiento del script consta de:
1.  **Carga y Normalización:** Descarga o lectura local del flujo relativo de los archivos FITS.
2.  **Limpieza de Datos:** Remoción de datos atípicos (*outliers*) mediante *Sigma Clipping*.
3.  **Aplanado por Órbitas:** Segmentación de los datos en órbitas independientes y aplanado local empleando el filtro de Savitzky-Golay.
4.  **Análisis de Frecuencias:** Bucle de prewhitening global y ajuste simultáneo multisenoidal de las frecuencias estelares significativas (criterio de Breger, $S/N \ge 4.0$, utilizando el formalismo local de Saesen et al. 2010).

> [!WARNING]
> **Preprocesamiento de datos de entrada (Remoción de transitorios):**
> Para garantizar un funcionamiento óptimo del pipeline, se recomienda **eliminar manualmente de los archivos FITS de entrada los picos transitorios de gran amplitud (de tipo delta)** de origen instrumental. Al tratarse de anomalías de muy corta duración pero de extrema amplitud, su presencia puede sesgar gravemente el ajuste polinómico de Savitzky-Golay e introducir frecuencias espurias dominantes en el análisis de Fourier posterior.

## Requisitos de instalación

Para ejecutar el script es necesario contar con Python 3 y las siguientes librerías:

```bash

pip install numpy pandas matplotlib scipy astropy lightkurve

```

## Instrucciones de uso

1. Clona o descarga este repositorio en tu equipo.
2. Crea una carpeta llamada `Datos_sector` en el mismo directorio del script y deposita en ella los archivos FITS de las curvas de luz a analizar.
3. Ejecuta el pipeline desde tu terminal:

```bash

python ClearcurvesSG_FrequencyAnalysis_sector.py

```

4. Los resultados (gráficos de diagnóstico, curvas limpias en texto plano, tablas de frecuencias en CSV y un resumen del descarte de puntos de la primera fase del script) se guardarán automáticamente en la carpeta autogenerada `Resultados_sector/`.
