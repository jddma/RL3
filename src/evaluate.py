"""Evalua el modelo congelado y los baselines por segmento.

Cada segmento empieza con el mismo capital y posicion cero.
No se concatenan sus resultados como una inversion continua.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .agent import DQNAgent
from .baselines import ComprarYMantener, AgenteAleatorio
from .environment import EntornoInversion

ROOT = Path(__file__).resolve().parents[1]


def evaluar_segmento(datos, politica, segmento, config):
    env = EntornoInversion(
        datos,
        capital_inicial=config["capital_inicial"],
        tasa_costo=config["tasa_costo"],
        recompensa=config["recompensa_final"],
        posicion_inicial=0.0,
    )
    obs, info = env.reset(
        seed=42,
        options={"segmento": segmento},
    )

    registros = [{
        "timestamp": info["timestamp"],
        "capital": info["capital"],
        "posicion": info["posicion"],
        "accion": np.nan,
        "posicion_objetivo": np.nan,
        "costo": 0.0,
        "rotacion": 0.0,
    }]

    while True:
        accion = politica.actuar_greedy(obs)
        obs, recompensa, terminado, truncado, info = env.step(accion)

        registros.append({
            "timestamp": info["timestamp"],
            "capital": info["capital"],
            "posicion": info["posicion"],
            "accion": accion,
            "posicion_objetivo": info["posicion_objetivo"],
            "costo": info["costo"],
            "rotacion": info["rotacion"],
        })

        if terminado or truncado:
            break

    trayectoria = pd.DataFrame(registros)
    capital = trayectoria["capital"].to_numpy()
    drawdown = capital / np.maximum.accumulate(capital) - 1
    decisiones = trayectoria.iloc[1:]

    metricas = {
        "segmento": segmento,
        "inicio": registros[0]["timestamp"],
        "fin": registros[-1]["timestamp"],
        "pasos": len(decisiones),
        "capital_inicial": capital[0],
        "capital_final": capital[-1],
        "retorno_acumulado": capital[-1] / capital[0] - 1,
        "max_drawdown": float(drawdown.min()),
        "costos_totales": info["costo_total"],
        "operaciones": info["operaciones"],
        "rotacion_total": decisiones["rotacion"].sum(),
        "proporcion_accion_0": (decisiones["accion"] == 0).mean(),
        "proporcion_accion_50": (decisiones["accion"] == 1).mean(),
        "proporcion_accion_100": (decisiones["accion"] == 2).mean(),
        "exposicion_real_media": decisiones["posicion"].mean(),
    }
    trayectoria["drawdown"] = drawdown
    return trayectoria, metricas


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--particion",
        choices=["validacion", "heldout"],
        default="validacion",
    )
    args = parser.parse_args()

    carpeta_modelo = ROOT / "modelo_congelado"
    checkpoint = carpeta_modelo / "modelo_final.pt"
    config = json.loads(
        (carpeta_modelo / "configuracion_final.json").read_text(
            encoding="utf-8"
        )
    )

    huella = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    if huella != config["sha256_checkpoint"]:
        raise ValueError("El checkpoint no coincide con la huella congelada")

    datos = pd.read_csv(
        ROOT / "resultados_corregidos" / f"{args.particion}.csv",
        index_col=0,
        parse_dates=True,
    )

    entorno = EntornoInversion(datos)
    agente = DQNAgent(
        entorno.observation_space.shape[0],
        entorno.action_space.n,
    )
    agente.cargar(checkpoint)

    salida = ROOT / f"evaluacion_{args.particion}"
    salida.mkdir(exist_ok=False)

    resultados = []

    for segmento in range(len(entorno.segmentos)):
        # Semillas predeterminadas, independientes por segmento.
        semilla_aleatoria = 42 + segmento
        politicas = {
            "DQN": agente,
            "Comprar_y_mantener": ComprarYMantener(),
            "Aleatorio": AgenteAleatorio(semilla_aleatoria),
        }

        for nombre, politica in politicas.items():
            trayectoria, metricas = evaluar_segmento(
                datos, politica, segmento, config
            )
            metricas["estrategia"] = nombre
            metricas["semilla_aleatoria"] = (
                semilla_aleatoria if nombre == "Aleatorio" else None
            )
            resultados.append(metricas)

            trayectoria.to_csv(
                salida / f"trayectoria_{nombre}_segmento_{segmento}.csv",
                index=False,
            )

    tabla = pd.DataFrame(resultados)
    tabla.to_csv(salida / "metricas.csv", index=False)

    protocolo = {
        "particion": args.particion,
        "sha256_checkpoint": huella,
        "capital_inicial_por_segmento": config["capital_inicial"],
        "tasa_costo": config["tasa_costo"],
        "posicion_inicial_por_segmento": 0.0,
        "semilla_aleatoria_por_segmento": "42 + indice del segmento",
        "tratamiento_segmentos": (
            "Evaluaciones independientes con reinicio de capital y "
            "posicion; sin liquidacion forzada al terminar."
        ),
    }
    (salida / "protocolo.json").write_text(
        json.dumps(protocolo, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(tabla[[
        "estrategia",
        "segmento",
        "pasos",
        "retorno_acumulado",
        "max_drawdown",
        "costos_totales",
        "operaciones",
    ]].to_string(index=False))
    print("\nResultados guardados en:", salida)


if __name__ == "__main__":
    main()