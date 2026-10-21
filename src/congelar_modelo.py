"""Reproduce la congelación desde los resultados de validación."""

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd


RAIZ = Path(__file__).resolve().parents[1]

JUSTIFICACION = (
    "Retorno medio practicamente igual al log, menor dispersion entre "
    "semillas, menor caida promedio y menores costos promedio. "
    "Sin afirmacion de superioridad estadistica."
)


def sha256(ruta):
    return hashlib.sha256(ruta.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--origen",
        type=Path,
        default=RAIZ / "resultados_comparacion_recompensas",
    )
    parser.add_argument(
        "--datos",
        type=Path,
        default=RAIZ / "resultados_corregidos",
    )
    parser.add_argument(
        "--salida",
        type=Path,
        default=RAIZ / "modelo_congelado_reproducido",
    )
    args = parser.parse_args()

    if args.salida.exists():
        raise FileExistsError(
            f"La carpeta de salida ya existe: {args.salida}"
        )

    configuracion = json.loads(
        (args.origen / "configuracion.json").read_text(encoding="utf-8")
    )
    resumen_datos = args.datos / "resumen.json"
    if not resumen_datos.is_file():
        raise FileNotFoundError(resumen_datos)

    resultados = pd.read_csv(args.origen / "comparacion_A_B.csv")

    # La recompensa simple fue elegida en la comparación A/B documentada.
    # Aquí se reproduce únicamente la selección de su semilla.
    grupo = resultados.loc[
        resultados["recompensa"].eq("simple")
    ].copy()

    if grupo.empty:
        raise ValueError("No hay resultados de recompensa simple.")

    columnas = ["semilla", "retorno", "max_drawdown"]
    if not np.isfinite(grupo[columnas].to_numpy(dtype=float)).all():
        raise ValueError("Hay métricas o semillas inválidas.")

    if grupo["semilla"].duplicated().any():
        raise ValueError("Hay semillas duplicadas para recompensa simple.")

    if not grupo["semilla"].eq(grupo["semilla"].astype(int)).all():
        raise ValueError("Las semillas deben ser enteras.")

    if set(grupo["semilla"].astype(int)) != set(configuracion["semillas"]):
        raise ValueError("Las semillas no coinciden con la configuración.")

    limite = 1.15 * grupo["max_drawdown"].abs().median()
    candidatas = grupo.loc[
        grupo["max_drawdown"].abs() <= limite
    ]

    if candidatas.empty:
        raise ValueError("Ninguna semilla cumple el límite de drawdown.")

    elegida = candidatas.sort_values(
        ["retorno", "semilla"],
        ascending=[False, True],
    ).iloc[0]

    semilla = int(elegida["semilla"])
    checkpoint = args.origen / f"dqn_simple_{semilla}.pt"
    huella = sha256(checkpoint)

    # Convertir los tipos de pandas a tipos JSON.
    metricas = json.loads(elegida.to_json())
    metricas["semilla"] = semilla

    try:
        origen_relativo = str(checkpoint.resolve().relative_to(RAIZ))
    except ValueError:
        origen_relativo = str(checkpoint.resolve())

    configuracion.update(
        {
            "recompensa": "simple",
            "recompensa_final": "simple",
            "semilla_congelada": semilla,
            "metricas_validacion": metricas,
            "checkpoint_origen": origen_relativo,
            "sha256_checkpoint": huella,
            "regla_seleccion": (
                "Mayor retorno en validacion entre semillas de recompensa "
                "simple con magnitud del drawdown <= 1.15 veces "
                "la mediana del grupo."
            ),
            "justificacion_recompensa": JUSTIFICACION,
            "heldout_utilizado_para_seleccion": False,
        }
    )

    texto_configuracion = json.dumps(
        configuracion,
        indent=2,
        ensure_ascii=False,
        allow_nan=False,
    )

    args.salida.mkdir(parents=True, exist_ok=False)
    destino = args.salida / "modelo_final.pt"
    shutil.copy2(checkpoint, destino)

    if sha256(destino) != huella:
        raise RuntimeError("El checkpoint copiado no coincide con el original.")

    shutil.copy2(resumen_datos, args.salida / "resumen_datos.json")
    (args.salida / "configuracion_final.json").write_text(
        texto_configuracion,
        encoding="utf-8",
    )

    print(f"Modelo congelado: recompensa simple, semilla {semilla}")
    print(f"Retorno de validacion: {elegida['retorno']:.2%}")
    print(f"SHA256: {huella}")
    print(f"Carpeta: {args.salida}")


if __name__ == "__main__":
    main()