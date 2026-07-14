# =============================================================================
#         SCRIPT DE PROCESAMIENTO EN LOTE (BATCH PROCESSING)
# PROCESA TODAS LAS ESTRELLAS EN Datos_sector Y GUARDA EN Resultados_sector
# =============================================================================

import os
import re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import lightkurve as lk
from astropy.table import Table
from astropy.io import fits
from astropy.timeseries import LombScargle
from scipy.optimize import curve_fit
import warnings

# Ignorar advertencias específicas de scipy.optimize y astropy
warnings.filterwarnings("ignore", module="scipy.optimize")
warnings.filterwarnings("ignore", category=UserWarning)

# =============================================================================
# PARÁMETROS CONFIGURABLES
# =============================================================================
DIR_DATOS = "Datos_sector"
DIR_RESULTADOS = "Resultados_sector"

# Carpetas internas (Opción A)
SUBDIR_CURVAS = os.path.join(DIR_RESULTADOS, "Curvas_Limpias")
SUBDIR_TABLAS = os.path.join(DIR_RESULTADOS, "Tablas_Frecuencias")
SUBDIR_GRAFICOS = os.path.join(DIR_RESULTADOS, "Graficos_Diagnostico")

# 1. Parámetros de Limpieza y Normalización
SIGMA_CLIPPING = 4.0        # Desviación estándar para remover outliers
APLICAR_SG = True          # Aplicar filtro Savitzky-Golay (flatten)
VENTANA_SG_DIAS = 2.0       # Ventana de aplanado de Savitzky-Golay en días
RECORTAR_BORDES = True      # Cortar los inicios y finales de observación (órbitas)
RECORTE_BORDES_DIAS = 0.3   # Días a recortar en cada borde de cada órbita (típicamente 0.3 a 0.5 días)

# 2. Parámetros de Análisis de Frecuencias (Prewhitening)
MAX_ITER = 30               # Límite máximo de iteraciones
SNR_UMBRAL = 4.0            # Criterio de parada clásico (Breger et al. 1993)
USAR_PARADA_SNR = True      # True para detenerse cuando S/N < SNR_UMBRAL, False para correr siempre MAX_ITER
ITER_LIMPIEZA = 2           # Frecuencias iniciales forzadas para limpiar derivas
FREQ_LIMITE_DERIVA = 0.35     # Frecuencias inferiores a este valor (en d^-1) son tratadas como deriva

global_resumen_limpieza = []  # Lista global para acumular los logs de descarte de puntos

# =============================================================================
# MODELOS Y FUNCIONES DE AJUSTE
# =============================================================================

def modelo_senoidal_simple(t, amplitud, frecuencia, fase, offset):
    """Modelo de una sola onda para estimaciones iniciales."""
    return amplitud * np.sin(2 * np.pi * frecuencia * t + fase) + offset

def modelo_multisenoidal(t, *params):
    """
    Modelo multisenoidal para ajuste global simultáneo (Improve All).
    Estructura de params: [offset, amp_1, freq_1, fase_1, amp_2, freq_2, fase_2, ...]
    """
    offset = params[0]
    y = np.zeros_like(t) + offset
    for j in range(1, len(params), 3):
        amp = params[j]
        freq = params[j+1]
        phase = params[j+2]
        y += amp * np.sin(2 * np.pi * freq * t + phase)
    return y

def evaluar_ruido_local_saesen(frecuencias, amplitudes, freq_pico):
    """Calcula la amplitud media en una ventana local según Saesen et al. (2010, Sect. 6.2)"""
    if freq_pico <= 3.0: 
        box_size = 1.0
    elif freq_pico <= 6.0: 
        box_size = 1.9
    elif freq_pico <= 11.0: 
        box_size = 3.9
    else: 
        box_size = 5.0
        
    mascara = (frecuencias > (freq_pico - box_size)) & (frecuencias < (freq_pico + box_size))
    mascara &= (np.abs(frecuencias - freq_pico) > 0.05)
    
    if not np.any(mascara): 
        return np.mean(amplitudes)
    return np.mean(amplitudes[mascara])

# =============================================================================
# EXTRACCIÓN DE METADATOS Y LECTURA
# =============================================================================

def extraer_metadatos_fits(ruta_fits):
    """
    Extrae el identificador de estrella y el sector desde el FITS o el nombre del archivo.
    Garantiza que los IDs de estrella (ej. SAP_2139_clean) se extraigan con prefijo 'WEBDA_'.
    """
    nombre_archivo = os.path.basename(ruta_fits)
    nombre_base = os.path.splitext(nombre_archivo)[0]
    
    object_name = None
    sector = None
    
    # Intentar leer desde las cabeceras FITS
    try:
        with fits.open(ruta_fits) as hdul:
            for hdu in hdul:
                header = hdu.header
                if not object_name:
                    object_name = header.get('OBJECT') or header.get('TICID')
                if not sector:
                    sector = header.get('SECTOR')
                if object_name and sector:
                    break
    except Exception as e:
        print(f"  [Advertencia] No se pudieron leer cabeceras de {nombre_archivo}: {e}")
        
    if object_name:
        object_name = str(object_name).strip()
        if object_name.isdigit():
            object_name = f"WEBDA_{object_name}"
        object_name = re.sub(r'[\s\-]+', '_', object_name)
    else:
        match_webda = re.search(r'(?:SAP|WEBDA|NGC\d+)_(\d+)', nombre_base, re.IGNORECASE) or re.search(r'\b(\d{3,5})\b', nombre_base)
        if match_webda:
            object_name = f"WEBDA_{match_webda.group(1)}"
        else:
            match_tic = re.search(r'\b(TIC|TICID)_?(\d+)\b', nombre_base, re.IGNORECASE) or re.search(r'\b(\d{7,10})\b', nombre_base)
            if match_tic:
                val = match_tic.group(2) if match_tic.lastindex else match_tic.group(0)
                object_name = f"TIC_{val}"
            else:
                object_name = nombre_base
                if object_name.isdigit():
                    object_name = f"WEBDA_{object_name}"
                object_name = re.sub(r'[\s\-]+', '_', object_name)
            
    # Procesar/Limpiar número de sector
    if sector is not None:
        sector = str(sector).strip()
    else:
        # Intentar extraer del nombre del archivo
        match_sec = re.search(r'-s(\d+)-', nombre_base, re.IGNORECASE) or re.search(r's(\d+)', nombre_base, re.IGNORECASE)
        if match_sec:
            sector = str(int(match_sec.group(1)))
        else:
            sector = "X"
            
    return object_name, sector

def cargar_curva_fits(ruta_fits):
    """
    Lee las columnas de tiempo, flujo y error de un FITS mediante astropy.table.
    """
    try:
        datos = Table.read(ruta_fits)
        colnames = datos.colnames
        
        # Buscar columnas prioritarias para tiempo, flujo y error
        col_time = next((c for c in colnames if c.upper() in ['TIME', 'BJD_TIME', 'BJD']), None)
        # Priorizar SAP_FLUX por petición del usuario (SAPhy)
        col_flux = next((c for c in colnames if c.upper() in ['SAP_FLUX', 'FLUX', 'PDCSAP_FLUX']), None)
        col_err = next((c for c in colnames if c.upper() in ['SAP_FLUX_ERR', 'FLUX_ERR', 'ERROR', 'ERROR_FLUX']), None)
        
        if col_time is None or col_flux is None:
            print(f"  [Error] No se hallaron columnas de Tiempo/Flujo en: {os.path.basename(ruta_fits)}")
            print(f"  Columnas disponibles: {colnames}")
            return None
            
        tiempo = datos[col_time].value if hasattr(datos[col_time], 'value') else datos[col_time]
        flujo = datos[col_flux].value if hasattr(datos[col_flux], 'value') else datos[col_flux]
        
        if col_err is not None:
            errores = datos[col_err].value if hasattr(datos[col_err], 'value') else datos[col_err]
        else:
            errores = np.ones_like(tiempo) * np.std(flujo) * 0.1
            
        # Remover filas con NaNs iniciales
        mask_nan = np.isnan(tiempo) | np.isnan(flujo)
        tiempo = tiempo[~mask_nan]
        flujo = flujo[~mask_nan]
        errores = errores[~mask_nan]
        
        lc = lk.LightCurve(time=tiempo, flux=flujo, flux_err=errores)
        return lc
    except Exception as e:
        print(f"  [Error] No se pudo leer el archivo FITS {os.path.basename(ruta_fits)}: {e}")
        return None

# =============================================================================
# PIPELINE DE PROCESAMIENTO COMPLETO POR ESTRELLA
# =============================================================================

def procesar_estrella(ruta_fits):
    global global_resumen_limpieza
    nombre_archivo = os.path.basename(ruta_fits)
    print("\n" + "-"*60)
    print(f" PROCESANDO: {nombre_archivo}")
    print("-"*60)
    
    # 1. Extraer metadatos
    object_name, sector = extraer_metadatos_fits(ruta_fits)
    
    # 2. Cargar curva de luz
    lc_bruta = cargar_curva_fits(ruta_fits)
    if lc_bruta is None:
        print("  [Error] Saltando estrella por error de lectura.")
        return
        
    puntos_iniciales = len(lc_bruta)
    if puntos_iniciales < 10:
        print("  [Error] Curva de luz vacía o con muy pocos puntos. Saltando.")
        return

    # Si el sector no se encontró, lo estimamos automáticamente mediante el tiempo medio en BTJD
    if sector == "X":
        t_mean = np.mean(lc_bruta.time.value)
        # TESS Sector 1 comenzó en BTJD ~1325, cada sector dura ~27.4 días
        sector = str(round((t_mean - 1325.0) / 27.4) + 1)
        print(f"  Identificador: {object_name} | Sector Estimado (BTJD medio {t_mean:.2f}): {sector}")
    else:
        print(f"  Identificador: {object_name} | Sector: {sector}")
        
    # 3. Limpieza y Normalización (Segmentación por órbitas para evitar gaps)
    print("  [1/2] Limpieza, Normalización y Aplanado...")
    lc_sin_nans = lc_bruta.remove_nans().normalize()
    puntos_sin_nans = len(lc_sin_nans)
    lc_limpia = lc_sin_nans.remove_outliers(sigma=SIGMA_CLIPPING)
    puntos_outliers = puntos_sin_nans - len(lc_limpia)
    
    # Identificar segmentos continuos de observación (gaps > 0.5 días indican perigeo/downlink)
    tiempo_limpio = lc_limpia.time.value
    diferencias = np.diff(tiempo_limpio)
    indices_gap = np.where(diferencias > 0.5)[0]
    
    limites_indices = [0]
    for idx in indices_gap:
        limites_indices.append(idx + 1)
    limites_indices.append(len(lc_limpia))
    
    segmentos_procesados = []
    puntos_recortados_bordes = 0
    
    for k in range(len(limites_indices) - 1):
        idx_inicio = limites_indices[k]
        idx_fin = limites_indices[k+1]
        lc_seg = lc_limpia[idx_inicio:idx_fin]
        
        if len(lc_seg) < 10:
            continue
            
        # Aplanar el segmento de órbita individualmente
        if APLICAR_SG:
            tiempo_seg = lc_seg.time.value
            delta_t_seg = tiempo_seg[-1] - tiempo_seg[0]
            cadencia_seg = delta_t_seg / len(lc_seg)
            
            ventana = int(VENTANA_SG_DIAS / cadencia_seg)
            if ventana % 2 == 0: 
                ventana += 1
            ventana = max(5, min(ventana, len(lc_seg) - 1))
            
            lc_seg_flat = lc_seg.flatten(window_length=ventana)
        else:
            lc_seg_flat = lc_seg.copy()
            lc_seg_flat.flux = lc_seg_flat.flux - 1.0
            
        if RECORTAR_BORDES:
            t_seg = lc_seg_flat.time.value
            t_min = t_seg[0]
            t_max = t_seg[-1]
            
            recorte_inicio = t_min + RECORTE_BORDES_DIAS
            recorte_fin = t_max - RECORTE_BORDES_DIAS
            
            mascara_conservar = (t_seg >= recorte_inicio) & (t_seg <= recorte_fin)
            puntos_antes = len(lc_seg_flat)
            lc_seg_flat = lc_seg_flat[mascara_conservar]
            puntos_despues = len(lc_seg_flat)
            puntos_recortados_bordes += (puntos_antes - puntos_despues)
            
            print(f"        Órbita {k+1} [{t_min:.2f} a {t_max:.2f}]: Recortados {puntos_antes - puntos_despues} puntos ({RECORTE_BORDES_DIAS} días por borde).")
            
        segmentos_procesados.append(lc_seg_flat)
        
    if not segmentos_procesados:
        print("  [Error] Todos los segmentos de órbita quedaron vacíos tras la limpieza. Saltando.")
        return
        
    # Re-ensamblar los segmentos limpios
    tiempos_cat = np.concatenate([s.time.value for s in segmentos_procesados])
    fluxes_cat = np.concatenate([s.flux.value for s in segmentos_procesados])
    errors_cat = np.concatenate([s.flux_err.value for s in segmentos_procesados])
    
    lc_final = lk.LightCurve(time=tiempos_cat, flux=fluxes_cat, flux_err=errors_cat)
    
    # Centrar la curva de luz completa y final en 0.0 restando su media
    lc_final.flux = lc_final.flux - np.mean(lc_final.flux)
    
    puntos_finales = len(lc_final)
    descarte_total = 100 * (1 - puntos_finales / puntos_iniciales)
    print(f"        Puntos finales: {puntos_finales} de {puntos_iniciales} ({descarte_total:.2f}% descartados en total, incl. {puntos_recortados_bordes} en bordes)")
        
    # Guardar curva limpia
    ruta_curva_txt = os.path.join(SUBDIR_CURVAS, f"{object_name}_S{sector}_SAPhy.txt")
    tiempo = lc_final.time.value
    flujo = lc_final.flux.value
    error = lc_final.flux_err.value if lc_final.flux_err is not None else np.ones_like(tiempo) * np.std(flujo) * 0.1
    
    np.savetxt(ruta_curva_txt, np.c_[tiempo, flujo, error], 
               fmt='%.6f', delimiter='\t', header='Tiempo\tFlujo\tError')
    print(f"        --> Curva limpia guardada en: {ruta_curva_txt}")
    
    # Registrar log de descarte de datos
    puntos_nans = puntos_iniciales - puntos_sin_nans
    info_estrella = (
        f"Estrella: {object_name} (Sector {sector})\n"
        f"  - Archivo original: {nombre_archivo}\n"
        f"  - Puntos iniciales (FITS): {puntos_iniciales}\n"
        f"  - Puntos removidos por NaNs: {puntos_nans}\n"
        f"  - Puntos removidos por Sigma Clipping (outliers): {puntos_outliers}\n"
        f"  - Puntos recortados en bordes de órbita: {puntos_recortados_bordes}\n"
        f"  - Puntos finales limpios: {puntos_finales}\n"
        f"  - Porcentaje total de descarte: {descarte_total:.2f}%\n"
        f"----------------------------------------\n"
    )
    global_resumen_limpieza.append(info_estrella)
    
    # 5. Análisis de Frecuencias (Prewhitening)
    print("  [2/2] Análisis de Frecuencias (Prewhitening)...")
    flujo_residual = flujo.copy()
    errores_flujo = error.copy()
    
    frecuencias_detectadas = []
    amplitudes_detectadas = []
    fases_detectadas = []
    offset_global = np.mean(flujo)
    
    resultados = []
    
    for i in range(MAX_ITER):
        ls = LombScargle(tiempo, flujo_residual, errores_flujo)
        frecuencias, potencias = ls.autopower(minimum_frequency=0.01, maximum_frequency=30.0)
        amplitudes = np.sqrt(2) * np.std(flujo_residual) * np.sqrt(potencias)
        
        from scipy.signal import find_peaks
        idx_picos, _ = find_peaks(amplitudes, distance=10)
        
        if len(idx_picos) == 0:
            print("        [Aviso] No se encontraron más picos locales en el espectro. Deteniendo.")
            break
            
        freqs_picos = frecuencias[idx_picos]
        amps_picos = amplitudes[idx_picos]
        
        # Calcular la relación S/N para cada pico local
        snrs = []
        for fp, ap in zip(freqs_picos, amps_picos):
            ruido_p = evaluar_ruido_local_saesen(frecuencias, amplitudes, fp)
            snrs.append(ap / ruido_p)
        snrs = np.array(snrs)
        
        idx_mejor = np.argmax(snrs)
        freq_pico = freqs_picos[idx_mejor]
        ruido_fisico = evaluar_ruido_local_saesen(frecuencias, amplitudes, freq_pico)
        
        amp_guess = amps_picos[idx_mejor]
        fase_guess = 0.0
        
        # Añadir al ajuste global
        frecuencias_detectadas.append(freq_pico)
        amplitudes_detectadas.append(amp_guess)
        fases_detectadas.append(fase_guess)
        
        # Parámetros iniciales
        p0 = [offset_global]
        for j in range(len(frecuencias_detectadas)):
            p0.extend([amplitudes_detectadas[j], frecuencias_detectadas[j], fases_detectadas[j]])
            
        # Bounds para curve_fit
        lower_bounds = [-np.inf]
        upper_bounds = [np.inf]
        for j in range(len(frecuencias_detectadas)):
            f_est = frecuencias_detectadas[j]
            if f_est < FREQ_LIMITE_DERIVA:
                f_min = max(0.001, f_est - 1e-4)
                f_max = f_est + 1e-4
            else:
                f_min = max(0.01, f_est - 0.002)
                f_max = min(30.0, f_est + 0.002)
            lower_bounds.extend([0.0, f_min, -np.inf])
            upper_bounds.extend([np.inf, f_max, np.inf])
            
        try:
            popt, pcov = curve_fit(modelo_multisenoidal, tiempo, flujo, p0=p0, 
                                   sigma=errores_flujo, absolute_sigma=True,
                                   bounds=(lower_bounds, upper_bounds))
            errores_ajuste = np.sqrt(np.diag(pcov))
        except RuntimeError:
            print(f"        Iter {i+1:02d}: Ajuste simultáneo no convergió. Deteniendo prewhitening.")
            break
            
        offset_global = popt[0]
        frecuencias_detectadas = []
        amplitudes_detectadas = []
        fases_detectadas = []
        
        for j in range(1, len(popt), 3):
            amplitudes_detectadas.append(np.abs(popt[j]))
            frecuencias_detectadas.append(popt[j+1])
            fases_detectadas.append(popt[j+2])
            
        freq_opt = frecuencias_detectadas[-1]
        amp_opt = amplitudes_detectadas[-1]
        fase_opt = fases_detectadas[-1]
        err_freq = errores_ajuste[-2]
        err_amp = errores_ajuste[-3]
        err_fase = errores_ajuste[-1]
        
        snr_final = amp_opt / ruido_fisico
        es_deriva = freq_opt < FREQ_LIMITE_DERIVA
        
        # Clasificar e identificar criterio de parada
        if es_deriva:
            estado = f"Deriva/Ruido Rojo (< {FREQ_LIMITE_DERIVA} d^-1)"
            print(f"        Iter {i+1:02d} | Freq: {freq_opt:.4f} d^-1 | S/N: {snr_final:.2f} -> {estado}")
        elif i < ITER_LIMPIEZA:
            estado = "Limpieza Forzada"
            print(f"        Iter {i+1:02d} | Freq: {freq_opt:.4f} d^-1 | S/N: {snr_final:.2f} -> {estado}")
        else:
            if USAR_PARADA_SNR and snr_final < SNR_UMBRAL:
                print(f"        [Parada] Iter {i+1:02d}: S/N = {snr_final:.2f} < {SNR_UMBRAL} (Breger)")
                break
            estado = "Modo Pulsación Estelar"
            print(f"        Iter {i+1:02d} | Freq: {freq_opt:.4f} d^-1 | S/N: {snr_final:.2f} -> {estado}")
            
        # Generar residuos
        modelo_y = modelo_multisenoidal(tiempo, *popt)
        flujo_residual = flujo - modelo_y
        
        # Graficar iteración actual
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4))
        zoom = tiempo < (tiempo[0] + 5)
        ax1.scatter(tiempo[zoom], flujo[zoom], color='gray', s=2, alpha=0.3, label='Original')
        
        # Graficar modelo_y segmentado para evitar la línea diagonal en el gap
        t_zoom = tiempo[zoom]
        y_zoom = modelo_y[zoom]
        gaps = np.where(np.diff(t_zoom) > 0.5)[0]
        lims = [0] + [g + 1 for g in gaps] + [len(t_zoom)]
        
        first = True
        for k in range(len(lims) - 1):
            t_seg = t_zoom[lims[k]:lims[k+1]]
            y_seg = y_zoom[lims[k]:lims[k+1]]
            if len(t_seg) > 0:
                ax1.plot(t_seg, y_seg, color='red', linewidth=2, label='Modelo Global' if first else "")
                first = False
        ax1.set_title(f"Ajuste Simultáneo - Iteración {i+1}")
        ax1.set_xlabel("Tiempo (Días)"); ax1.set_ylabel("Flujo Relativo")
        ax1.legend()
        
        ax2.plot(frecuencias, amplitudes, color='black', alpha=0.7, label='Residuo')
        ax2.axvline(freq_opt, color='red', linestyle='--', label=f'Pico: {freq_opt:.3f}')
        ax2.set_xlim(0, max(10, freq_opt + 2))
        ax2.set_title(f"Periodograma - S/N = {snr_final:.2f}")
        ax2.set_xlabel("Frecuencia (d^-1)"); ax2.set_ylabel("Amplitud")
        ax2.legend()
        plt.tight_layout()
        
        ruta_grafico = os.path.join(SUBDIR_GRAFICOS, f"{object_name}_S{sector}_Periodograma_I{i+1}.png")
        fig.savefig(ruta_grafico, dpi=300)
        plt.close(fig)
        
        resultados.append({
            'Iteración': i + 1,
            'Tipo': estado,
            'Frecuencia_d-1': round(freq_opt, 5),
            'Err_Frecuencia': round(err_freq, 6),
            'Amplitud': round(amp_opt, 5),
            'Err_Amplitud': round(err_amp, 6),
            'Fase_rad': round(fase_opt, 5),
            'Err_Fase': round(err_fase, 6),
            'S/N': round(snr_final, 2)
        })
        
    if resultados:
        df_res = pd.DataFrame(resultados)
        ruta_tabla = os.path.join(SUBDIR_TABLAS, f"{object_name}_S{sector}_FreqTabPhy.csv")
        df_res.to_csv(ruta_tabla, index=False)
        print(f"        --> Tabla de frecuencias guardada en: {ruta_tabla}")
    else:
        print("        [Advertencia] No se extrajeron frecuencias para esta estrella.")

# =============================================================================
# ESTRUCTURA PRINCIPAL DE EJECUCIÓN
# =============================================================================

def ejecutar_procesamiento_lote():
    print("="*60)
    print(" INICIANDO BATCH PIPELINE DE ANÁLISIS DE ESTRELLAS")
    print("="*60)
    
    # 1. Crear directorios
    os.makedirs(DIR_DATOS, exist_ok=True)
    os.makedirs(SUBDIR_CURVAS, exist_ok=True)
    os.makedirs(SUBDIR_TABLAS, exist_ok=True)
    os.makedirs(SUBDIR_GRAFICOS, exist_ok=True)
    
    # 2. Buscar archivos FITS
    archivos = [f for f in os.listdir(DIR_DATOS) if f.lower().endswith(('.fits', '.fit'))]
    
    if not archivos:
        print(f"\n[Aviso] No se encontraron archivos FITS (.fits o .fit) en '{DIR_DATOS}'.")
        print(f"Por favor, deposita tus curvas de luz FITS en: '{os.path.abspath(DIR_DATOS)}'")
        return
        
    print(f"\nSe encontraron {len(archivos)} archivo(s) para procesar.")
    
    for archivo in archivos:
        ruta_fits = os.path.join(DIR_DATOS, archivo)
        procesar_estrella(ruta_fits)
        
    if global_resumen_limpieza:
        ruta_resumen = os.path.join(DIR_RESULTADOS, "resumen_limpieza.txt")
        with open(ruta_resumen, "w", encoding="utf-8") as f:
            f.write("============================================================\n")
            f.write(" RESUMEN DE LIMPIEZA Y DESCARTE DE PUNTOS POR ESTRELLA\n")
            f.write("============================================================\n\n")
            f.write("".join(global_resumen_limpieza))
        print(f"\n --> Resumen de limpieza general guardado en: {os.path.abspath(ruta_resumen)}")
        
    print("\n" + "="*60)
    print(" BATCH PIPELINE COMPLETO - RESULTADOS ALMACENADOS EN:")
    print(f" --> {os.path.abspath(DIR_RESULTADOS)}")
    print("="*60)

if __name__ == "__main__":
    ejecutar_procesamiento_lote()
