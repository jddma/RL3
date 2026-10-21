"""Entrenamiento del agente DQN sobre el entorno de inversion.

Este script entrena y congela el modelo final que se usara en una evaluacion
posterior sobre ``heldout.csv`` (fuera del alcance de este script: aqui solo
se entrena y valida con ``train.csv`` / ``validacion.csv``).

Reutiliza los hiperparametros ya validados en ``experimento.py`` (misma
arquitectura, mismo calendario de epsilon, mismos costos y capital inicial)
para mantener continuidad con los resultados ya reportados en ``README.md``.
La recompensa final elegida para el modelo congelado es **A (retorno simple
neto)**.

Uso (desde la raiz del proyecto):
    python -m src.train                     # entrena 3 semillas, valida y congela
    python -m src.train --pasos 2000         # corrida rapida de verificacion
    python -m src.train --semillas 42        # entrenar solo una semilla
    python -m src.train --no-congelar         # entrenar/validar sin seleccionar modelo final

Semillas fijadas: ``random``, ``numpy``, ``torch`` y ``env.action_space``.
Advertencia de no determinismo: en CPU con las mismas semillas los resultados
son reproducibles; en GPU, algunas operaciones de cuDNN no son deterministas
salvo que se fuerce ``torch.use_deterministic_algorithms(True)`` (no usado
aqui porque el entrenamiento corre en CPU).
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch

try:
    from .agent import ConfiguracionDQN, DQNAgent
    from .environment import EntornoInversion
except ImportError:  # al ejecutar directamente desde src/
    from agent import ConfiguracionDQN, DQNAgent
    from environment import EntornoInversion

ROOT = Path(__file__).resolve().parents[1]
# AJUSTE RECOMENDADO
# RESULTADOS = ROOT / "resultados"
DATOS = ROOT / "resultados_corregidos"
RESULTADOS = ROOT / "resultados_dqn"
# FIN DE AJUSTE RECOMENDADO

# Hiperparametros ya congelados en experimento.py (ver README.md); se
# reutilizan tal cual para el modelo final de este entrenamiento.
CONFIG = dict(
    semillas=[42, 123, 2026],
    recompensa="simple",  # A: retorno simple neto (recomendacion del README)
    pasos=30_000,
    longitud_episodio=256,
    capital_inicial=100_000.0,
    tasa_costo=0.001,
    gamma=0.99,
    lr=0.0005,
    batch_size=64,
    buffer_size=20_000,
    inicio_aprendizaje=1_000,
    frecuencia_target=500,
    epsilon_inicial=1.0,
    epsilon_final=0.05,
    epsilon_decay=20_000,
)


def fijar_semillas(seed: int) -> None:
    """Fija las semillas de python ``random``, ``numpy`` y ``torch``."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def construir_agente(env: EntornoInversion, seed: int) -> DQNAgent:
    config = ConfiguracionDQN(
        gamma=CONFIG["gamma"], lr=CONFIG["lr"], batch_size=CONFIG["batch_size"],
        buffer_size=CONFIG["buffer_size"], inicio_aprendizaje=CONFIG["inicio_aprendizaje"],
        frecuencia_target=CONFIG["frecuencia_target"], epsilon_inicial=CONFIG["epsilon_inicial"],
        epsilon_final=CONFIG["epsilon_final"], epsilon_decay=CONFIG["epsilon_decay"], semilla=seed,
    )
    return DQNAgent(env.observation_space.shape[0], env.action_space.n, config)


def entrenar(datos: pd.DataFrame, recompensa: str, seed: int, pasos: int):
    """Entrena un agente DQN desde cero sobre ``datos`` (solo train.csv).

    Episodios de longitud fija con inicio aleatorio, restringidos a
    segmentos horarios continuos.

    Los limites artificiales de longitud permiten bootstrap cuando existe
    continuacion dentro del segmento. El final del segmento no permite
    bootstrap porque no hay una transicion horaria posterior disponible.

    Devuelve el agente entrenado y el historial de recompensa acumulada
    por episodio.
    """
    fijar_semillas(seed)
    env = EntornoInversion(
        datos, recompensa=recompensa, tasa_costo=CONFIG["tasa_costo"],
        capital_inicial=CONFIG["capital_inicial"], longitud_episodio=CONFIG["longitud_episodio"],
        inicio_aleatorio=True,
    )
    env.action_space.seed(seed)
    agente = construir_agente(env, seed)

    historial = []
    obs, _ = env.reset(seed=seed)
    recompensa_episodio = 0.0
    episodio = 0
    for paso in range(pasos):
        accion = agente.actuar(obs, paso)
    # AJUSTE RECOMENDADO 
    #    siguiente_obs, r, terminado, truncado, _info = env.step(accion)
    #    done = terminado or truncado
    #    agente.observar(obs, accion, r, siguiente_obs, done)
        siguiente_obs, r, terminado, truncado, _info = env.step(accion)

        # Reiniciar el episodio tanto por terminación como por truncamiento.
        done = terminado or truncado

        # No estimar continuación al final del segmento disponible.
        fin_segmento = env.segmentos[_info["segmento"]][1] if (
            "segmento" in _info
        ) else next(
            fin for inicio, fin in env.segmentos
            if inicio <= env._t <= fin
        )

        sin_continuacion = terminado or env._t >= fin_segmento

        # El límite artificial de 256 pasos permite bootstrap si hay
        # más observaciones horarias dentro del mismo segmento.
        agente.observar(
            obs, accion, r, siguiente_obs, sin_continuacion
        )
    # FIN DE AJUSTE RECOMENDADO    
        obs = siguiente_obs
        recompensa_episodio += r
        perdida = agente.aprender()
        if done:
            historial.append(dict(
                recompensa=recompensa, semilla=seed, episodio=episodio, paso=paso + 1,
                recompensa_acumulada=recompensa_episodio,
                perdida=perdida if perdida is not None else float("nan"),
                epsilon=agente.epsilon(paso),
            ))
            episodio += 1
            recompensa_episodio = 0.0
            obs, _ = env.reset()
    return agente, historial


def validar(datos: pd.DataFrame, agente: DQNAgent, recompensa: str, seed: int) -> dict:
    """Evalua con politica greedy (sin exploracion) sobre un episodio completo de ``datos``.

    Usado unicamente sobre ``validacion.csv``; el held-out permanece cerrado
    (se usara en la etapa de evaluacion, fuera de este alcance).
    """
    env = EntornoInversion(
        datos, recompensa=recompensa, tasa_costo=CONFIG["tasa_costo"],
        capital_inicial=CONFIG["capital_inicial"],
    )
    obs, _ = env.reset(seed=seed)
    curva = [CONFIG["capital_inicial"]]
    posiciones = []
    # AJUSTE RECOMENDADO
    acciones = []
    # FIN DE AJUSTE RECOMENDADO
    info = {}
    while True:
        accion = agente.actuar_greedy(obs)
        obs, _r, terminado, truncado, info = env.step(accion)
        curva.append(info["capital"])
        posiciones.append(info["posicion"])
        # AJUSTE RECOMENDADO
        acciones.append(accion)
        # FIN DE AJUSTE RECOMENDADO
        if terminado or truncado:
            break
    curva = np.array(curva)
    pos = np.array(posiciones)
    # AJUSTE RECOMENDADO
    acciones = np.array(acciones)
    # FIN DE AJUSTE RECOMENDADO
    maximos = np.maximum.accumulate(curva)
    return dict(
        recompensa=recompensa, semilla=seed,
        retorno=float(curva[-1] / curva[0] - 1),
        max_drawdown=float((curva / maximos - 1).min()),
        operaciones=int(info["operaciones"]),
        operaciones_por_1000_pasos=float(info["operaciones"] / len(pos) * 1000),
        costos_totales=float(info["costo_total"]),
        # AJUSTE RECOMENDADO 
        # exposicion_0=float(np.mean(pos == 0)),
        # exposicion_50=float(np.mean(pos == 0.5)),
        # exposicion_100=float(np.mean(pos == 1)),
        proporcion_accion_0=float(np.mean(acciones == 0)),
        proporcion_accion_50=float(np.mean(acciones == 1)),
        proporcion_accion_100=float(np.mean(acciones == 2)),
        exposicion_real_media=float(np.mean(pos)),
        # FIN DE AJUSTE RECOMENDADO
        capital_final=float(curva[-1]),
    )


def seleccionar_semilla_final(validaciones: list[dict]) -> dict:
    """Regla de seleccion del modelo final a congelar.

    Se elige la semilla con mayor retorno de validacion entre las que
    entrenaron con la recompensa final (A: simple), siempre que su maximo
    drawdown no sea peor que la mediana de drawdown del grupo (en valor
    absoluto). Esto evita congelar una semilla con retorno alto pero riesgo
    desproporcionado, siguiendo el criterio de "retorno condicionado a
    drawdown y estabilidad" ya usado al comparar las recompensas A y B. No se
    usa el held-out para esta decision.
    """
    if not validaciones:
        raise ValueError("No hay resultados de validacion para seleccionar una semilla")
    mediana_dd = float(np.median([abs(v["max_drawdown"]) for v in validaciones]))
    candidatas = sorted(validaciones, key=lambda v: v["retorno"], reverse=True)
    for v in candidatas:
        if abs(v["max_drawdown"]) <= mediana_dd * 1.15:
            return v
    return candidatas[0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recompensa", default=CONFIG["recompensa"], choices=["simple", "log"])
    parser.add_argument("--semillas", type=int, nargs="+", default=CONFIG["semillas"])
    parser.add_argument("--pasos", type=int, default=CONFIG["pasos"])
    parser.add_argument("--salida", type=Path, default=RESULTADOS)
    parser.add_argument("--congelar", dest="congelar", action="store_true", default=True)
    parser.add_argument("--no-congelar", dest="congelar", action="store_false")
    args = parser.parse_args()

    args.salida.mkdir(exist_ok=True)
    # AJUSTES RECOMENDADOS 
    # train = pd.read_csv(RESULTADOS / "train.csv", index_col=0, parse_dates=True)
    # val = pd.read_csv(RESULTADOS / "validacion.csv", index_col=0, parse_dates=True)
    train = pd.read_csv(
        DATOS / "train.csv",
        index_col=0,
        parse_dates=True,
    )
    val = pd.read_csv(
        DATOS / "validacion.csv",
        index_col=0,
        parse_dates=True,
    )
    # FIN DE AJUSTES RECOMENDADOS
     
    config_ejecucion = {**CONFIG, "recompensa": args.recompensa, "semillas": args.semillas, "pasos": args.pasos}
    (args.salida / "configuracion_agente.json").write_text(
        json.dumps(config_ejecucion, indent=2), encoding="utf-8"
    )

    historial_total, validaciones = [], []
    for seed in args.semillas:
        print(f"Entrenando recompensa={args.recompensa} semilla={seed} pasos={args.pasos}", flush=True)
        agente, historial = entrenar(train, args.recompensa, seed, args.pasos)
        historial_total.extend(historial)
        metricas = validar(val, agente, args.recompensa, seed)
        validaciones.append(metricas)
        agente.guardar(args.salida / f"dqn_{args.recompensa}_{seed}.pt")
        pd.DataFrame(historial_total).to_csv(args.salida / "historial_entrenamiento_agente.csv", index=False)
        pd.DataFrame(validaciones).to_csv(args.salida / "validacion_entrenamiento_agente.csv", index=False)
        print(f"  -> retorno={metricas['retorno']:.2%} max_drawdown={metricas['max_drawdown']:.2%}")

    if args.congelar:
        elegida = seleccionar_semilla_final(validaciones)
        checkpoint_origen = args.salida / f"dqn_{args.recompensa}_{elegida['semilla']}.pt"
        checkpoint_final = args.salida / "modelo_final.pt"
        checkpoint_final.write_bytes(checkpoint_origen.read_bytes())
        config_final = {
            **config_ejecucion,
            "semilla_congelada": elegida["semilla"],
            "metricas_validacion_congelada": elegida,
            "regla_seleccion": (
                "Mayor retorno de validacion entre semillas entrenadas con la "
                "recompensa final, con max_drawdown <= 1.15x la mediana de "
                "drawdown del grupo; held-out no utilizado."
            ),
        }
        (args.salida / "configuracion_final.json").write_text(
            json.dumps(config_final, indent=2), encoding="utf-8"
        )
        print(
            f"Modelo final congelado: semilla={elegida['semilla']} "
            f"retorno={elegida['retorno']:.2%} max_drawdown={elegida['max_drawdown']:.2%} "
            f"-> {checkpoint_final}"
        )


if __name__ == "__main__":
    main()
