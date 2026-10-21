"""Compara recompensas simple y log con el mismo protocolo.

Usa exclusivamente entrenamiento y validacion.
No consulta held-out ni congela un modelo final.

Ejecutar:
    python -m src.experimento
"""
from pathlib import Path
import json

import pandas as pd

from .train import CONFIG, entrenar, validar

ROOT = Path(__file__).resolve().parents[1]
DATOS = ROOT / "resultados_corregidos"
SALIDA = ROOT / "resultados_comparacion_recompensas"


def main():
    # Evita sobrescribir accidentalmente un experimento anterior.
    SALIDA.mkdir(exist_ok=False)

    train = pd.read_csv(
        DATOS / "train.csv", index_col=0, parse_dates=True
    )
    validacion = pd.read_csv(
        DATOS / "validacion.csv", index_col=0, parse_dates=True
    )

    configuracion = {
        **CONFIG,
        "recompensas_comparadas": ["simple", "log"],
        "datos": str(DATOS),
        "protocolo": (
            "Mismas semillas y pasos para ambas recompensas; "
            "entorno y datos corregidos; seleccion solo en validacion."
        ),
    }
    (SALIDA / "configuracion.json").write_text(
        json.dumps(configuracion, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    resultados = []
    historial = []

    for semilla in CONFIG["semillas"]:
        for recompensa in ("simple", "log"):
            print(
                f"Entrenando recompensa={recompensa} "
                f"semilla={semilla} pasos={CONFIG['pasos']}",
                flush=True,
            )

            agente, registros = entrenar(
                train, recompensa, semilla, CONFIG["pasos"]
            )
            metricas = validar(
                validacion, agente, recompensa, semilla
            )

            agente.guardar(
                SALIDA / f"dqn_{recompensa}_{semilla}.pt"
            )

            resultados.append(metricas)
            historial.extend(registros)

            pd.DataFrame(resultados).to_csv(
                SALIDA / "comparacion_A_B.csv", index=False
            )
            pd.DataFrame(historial).to_csv(
                SALIDA / "historial_entrenamiento.csv", index=False
            )

            print(
                f"  -> retorno={metricas['retorno']:.2%} "
                f"max_drawdown={metricas['max_drawdown']:.2%}",
                flush=True,
            )

    tabla = pd.DataFrame(resultados)
    columnas = [
        "retorno",
        "max_drawdown",
        "operaciones",
        "costos_totales",
        "proporcion_accion_0",
        "proporcion_accion_50",
        "proporcion_accion_100",
        "exposicion_real_media",
    ]

    resumen = tabla.groupby("recompensa")[columnas].agg(
        ["mean", "std"]
    )
    resumen.to_csv(SALIDA / "resumen_A_B.csv")

    print("\nComparacion completada:")
    print(resumen.to_string())


if __name__ == "__main__":
    main()
