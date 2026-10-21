"""Análisis exploratorio de los datos corregidos de entrenamiento."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.features import FEATURES


RAIZ = Path(__file__).resolve().parents[1]
DATOS = RAIZ / "resultados_corregidos"
SALIDA = RAIZ / "eda_actividad1"


def main():
    datos = pd.read_csv(
        DATOS / "train.csv",
        index_col=0,
        parse_dates=True,
    )
    datos.index = pd.to_datetime(datos.index, utc=True)

    resumen = json.loads(
        (DATOS / "resumen.json").read_text(encoding="utf-8")
    )

    columnas = ["open", "high", "low", "close", "volume"] + FEATURES

    if datos.empty:
        raise ValueError("El conjunto de entrenamiento está vacío.")

    if not datos.index.is_monotonic_increasing or not datos.index.is_unique:
        raise ValueError("Las fechas deben estar ordenadas y sin duplicados.")

    if datos.index.hasnans:
        raise ValueError("Hay fechas inválidas.")

    if not np.isfinite(datos[columnas].to_numpy(dtype=float)).all():
        raise ValueError("Hay valores vacíos o infinitos.")

    # Recuperar las unidades originales, sin ajustar otro escalador.
    for columna in FEATURES:
        media = resumen["media_entrenamiento"][columna]
        desviacion = resumen["desviacion_entrenamiento"][columna]
        if not np.isfinite(desviacion) or desviacion <= 0:
            raise ValueError(f"Desviación inválida: {columna}")
        datos[columna] = datos[columna] * desviacion + media

    diferencias = datos.index.to_series().diff()
    segmentos = diferencias.ne(pd.Timedelta(hours=1)).cumsum()

    # Verificar retornos donde hay dos filas consecutivas disponibles.
    continuas = diferencias.eq(pd.Timedelta(hours=1))
    retornos_comprobacion = datos["close"].pct_change(fill_method=None)

    if not np.allclose(
        datos.loc[continuas, "retorno_1h"],
        retornos_comprobacion.loc[continuas],
        rtol=1e-7,
        atol=1e-10,
    ):
        raise ValueError("Los retornos recuperados no coinciden con los precios.")

    # Evitar sobrescribir una ejecución anterior.
    SALIDA.mkdir(exist_ok=False)

    estadisticas = datos[columnas].describe(
        percentiles=[0.01, 0.05, 0.50, 0.95, 0.99]
    ).T
    estadisticas.to_csv(
        SALIDA / "estadisticas_entrenamiento.csv",
        index_label="variable",
    )

    tramos = []
    for numero, (_, tramo) in enumerate(datos.groupby(segmentos)):
        tramos.append(
            {
                "segmento": numero,
                "inicio": str(tramo.index[0]),
                "fin": str(tramo.index[-1]),
                "filas": len(tramo),
            }
        )

    pd.DataFrame(tramos).to_csv(
        SALIDA / "segmentos_entrenamiento.csv", index=False
    )

    def graficar_serie(columna, titulo, etiqueta, archivo, factor=1):
        fig, ax = plt.subplots(figsize=(11, 4))

        # Dibujar cada segmento aparte evita unir precios a través de huecos.
        for _, tramo in datos.groupby(segmentos):
            ax.plot(
                tramo.index,
                tramo[columna] * factor,
                linewidth=0.7,
            )

        ax.set_title(titulo)
        ax.set_xlabel("Fecha (UTC)")
        ax.set_ylabel(etiqueta)
        ax.grid(alpha=0.25)
        fig.autofmt_xdate()
        fig.tight_layout()
        fig.savefig(SALIDA / archivo, dpi=150)
        plt.close(fig)

    graficar_serie(
        "close",
        "Entrenamiento — precio de cierre BTC/USDT",
        "Precio (USDT)",
        "precio_entrenamiento.png",
    )
    graficar_serie(
        "retorno_1h",
        "Entrenamiento — retornos horarios",
        "Retorno (%)",
        "retornos_entrenamiento.png",
        factor=100,
    )
    graficar_serie(
        "volatilidad_24h",
        "Entrenamiento — volatilidad móvil de 24 horas",
        "Desviación estándar de retornos (%)",
        "volatilidad_entrenamiento.png",
        factor=100,
    )
    graficar_serie(
        "volume",
        "Entrenamiento — volumen reportado por la fuente",
        "Volumen",
        "volumen_entrenamiento.png",
    )

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(datos["retorno_1h"] * 100, bins=100, edgecolor="white")
    ax.set_title("Entrenamiento — distribución de retornos horarios")
    ax.set_xlabel("Retorno (%)")
    ax.set_ylabel("Frecuencia")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(SALIDA / "distribucion_entrenamiento.png", dpi=150)
    plt.close(fig)

    retornos = datos["retorno_1h"]
    diagnostico = {
        "particion": "train",
        "archivo": "resultados_corregidos/train.csv",
        "filas": len(datos),
        "inicio": str(datos.index[0]),
        "fin": str(datos.index[-1]),
        "segmentos": len(tramos),
        "retorno_horario_medio_pct": float(retornos.mean() * 100),
        "desviacion_retorno_horario_pct": float(retornos.std() * 100),
        "retorno_horario_minimo_pct": float(retornos.min() * 100),
        "retorno_horario_maximo_pct": float(retornos.max() * 100),
        "asimetria_retornos": float(retornos.skew()),
        "exceso_curtosis_retornos": float(retornos.kurt()),
        "nota": (
            "Indicadores recuperados con el escalado de entrenamiento. "
            "Los huecos entre filas retenidas incluyen el calentamiento "
            "descartado; no equivalen directamente a horas ausentes "
            "en la fuente. No se leyeron validación ni heldout."
        ),
    }
    (SALIDA / "resumen_eda.json").write_text(
        json.dumps(diagnostico, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )

    print(json.dumps(diagnostico, indent=2, ensure_ascii=False))
    print(f"\nResultados guardados en: {SALIDA}")


if __name__ == "__main__":
    main()