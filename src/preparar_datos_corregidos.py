"""Reconstruye datos corregidos desde los OHLCV de los CSV originales."""

import argparse
import json
from pathlib import Path

import pandas as pd

from src.features import preparar, indicadores, ajustar_escalado, escalar


RAIZ = Path(__file__).resolve().parents[1]
NOMBRES = ("train", "validacion", "heldout")
OHLCV = ["open", "high", "low", "close", "volume"]


def reconstruir(origen, salida):
    resumen_original = json.loads(
        (origen / "resumen.json").read_text(encoding="utf-8")
    )
    cortes = resumen_original["particiones"]

    limites = {}
    anteriores = None
    partes = []

    for nombre in NOMBRES:
        inicio = pd.to_datetime(cortes[nombre]["inicio"], utc=True)
        fin = pd.to_datetime(cortes[nombre]["fin"], utc=True)

        if inicio > fin or (anteriores is not None and inicio <= anteriores):
            raise ValueError("Los cortes originales se superponen o son inválidos.")

        limites[nombre] = (inicio, fin)
        anteriores = fin

        datos = pd.read_csv(
            origen / f"{nombre}.csv",
            index_col=0,
            parse_dates=True,
        )
        datos = preparar(datos[OHLCV])

        if (
            len(datos) != cortes[nombre]["filas"]
            or datos.index[0] != inicio
            or datos.index[-1] != fin
        ):
            raise ValueError(
                f"{nombre}: el CSV no coincide con el resumen original."
            )

        partes.append(datos)

    # Calcular sobre la serie cronológica completa permite usar historial
    # anterior al comienzo de validación y heldout, sin usar datos futuros.
    recuperados = preparar(pd.concat(partes))
    calculados = indicadores(recuperados)

    particiones = {}
    for nombre in NOMBRES:
        inicio, fin = limites[nombre]
        tramo = calculados.loc[
            (calculados.index >= inicio) & (calculados.index <= fin)
        ].copy()

        if len(tramo) < 2:
            raise ValueError(f"{nombre}: observaciones útiles insuficientes.")

        particiones[nombre] = tramo

    media, desviacion = ajustar_escalado(particiones["train"])

    descripcion = {}
    resultados = {}

    for nombre, tramo in particiones.items():
        resultados[nombre] = escalar(tramo, media, desviacion)

        numero_segmentos = int(
            tramo.index.to_series().diff()
            .ne(pd.Timedelta(hours=1))
            .sum()
        )
        descripcion[nombre] = {
            "filas": len(tramo),
            "inicio": str(tramo.index[0]),
            "fin": str(tramo.index[-1]),
            "segmentos": numero_segmentos,
        }

    resumen = {
        "fuente": "OHLCV recuperado de los tres CSV originales",
        "limitacion": (
            "No se recuperaron las 24 observaciones iniciales descartadas."
        ),
        "tratamiento_huecos": (
            "Indicadores independientes por segmento horario; "
            "24 filas de calentamiento por segmento."
        ),
        "cortes_originales": cortes,
        "particiones": descripcion,
        "media_entrenamiento": media.to_dict(),
        "desviacion_entrenamiento": desviacion.to_dict(),
    }

    # La carpeta debe ser nueva para preservar ejecuciones anteriores.
    salida.mkdir(parents=True, exist_ok=False)

    for nombre, datos in resultados.items():
        datos.to_csv(salida / f"{nombre}.csv", index_label="timestamp")

    (salida / "resumen.json").write_text(
        json.dumps(resumen, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )

    for nombre, datos in descripcion.items():
        print(
            f"{nombre}: {datos['filas']} filas, "
            f"{datos['segmentos']} segmentos"
        )
    print(f"\nArchivos guardados en: {salida}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--origen",
        type=Path,
        default=RAIZ / "resultados",
    )
    parser.add_argument(
        "--salida",
        type=Path,
        default=RAIZ / "resultados_corregidos_reproducidos",
    )
    args = parser.parse_args()

    if args.salida.exists():
        raise FileExistsError(
            f"La carpeta de salida ya existe: {args.salida}"
        )

    reconstruir(args.origen, args.salida)


if __name__ == "__main__":
    main()