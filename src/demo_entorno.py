"""Prueba el entorno con politicas fijas sobre el conjunto de VALIDACION.

No usa el held-out. Sirve para comprobar la dinamica de capital y mostrar el
efecto de los costos de transaccion antes de entrenar al agente.

Uso (desde la raiz del proyecto):
    python src/demo_entorno.py
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from environment import EntornoInversion

RAIZ = Path(__file__).resolve().parents[1]
CONFIG = {
    "capital_inicial": 100_000.0,
    "tasa_costo": 0.001,  # 10 puntos basicos
    "recompensa": "simple",
    "posicion_inicial": 0.0,
}
SEMILLA = 42


def simular(datos, politica, tasa_costo):
    """Corre un episodio completo y devuelve la curva de capital y el resumen."""
    env = EntornoInversion(
        datos,
        capital_inicial=CONFIG["capital_inicial"],
        tasa_costo=tasa_costo,
        recompensa=CONFIG["recompensa"],
        posicion_inicial=CONFIG["posicion_inicial"],
    )
# AJUSTE RECOMENDADO 
    if len(env.segmentos) != 1:
        raise ValueError(
            "Esta demostracion requiere un unico segmento continuo. "
            "La evaluacion de varios segmentos se implementa por separado."
        )
# FIN AJUSTE RECOMENDADO    
    env.reset(seed=SEMILLA)
    capital, posiciones = [CONFIG["capital_inicial"]], []
    paso, info = 0, {}
    while True:
        accion = politica(paso)
        _, _, terminado, truncado, info = env.step(accion)
        capital.append(info["capital"])
        posiciones.append(info["posicion"])
        paso += 1
        if terminado or truncado:
            break
    capital = np.array(capital)
    maximo = np.maximum.accumulate(capital)
    return capital, {
        "retorno_acumulado": capital[-1] / capital[0] - 1,
        "max_drawdown": float((capital / maximo - 1).min()),
        "costos_totales": info["costo_total"],
        "operaciones": info["operaciones"],
    }


def main():
# AJUSTE RECOMENDADO     
    #datos = pd.read_csv(RAIZ / "resultados" / "validacion.csv", index_col=0, parse_dates=True)
    datos = pd.read_csv(RAIZ / "resultados_corregidos" / "validacion.csv", index_col=0, parse_dates=True)
# FIN AJUSTE RECOMENDADO    
    politicas = {
        "Siempre 0 % (efectivo)": lambda t: 0,
# AJUSTE RECOMENDADO 
        # "Siempre 50 %": lambda t: 1,
        "Objetivo 50 % con rebalance horario": lambda t: 1,
# FIN AJUSTE RECOMENDADO        
        "Siempre 100 % (comprar y mantener)": lambda t: 2,
        "Alterna 0 % / 100 % cada hora": lambda t: 2 if t % 2 == 0 else 0,
    }
    tasa = CONFIG["tasa_costo"]
    filas, curvas = [], {}
    for nombre, pol in politicas.items():
        curva, res = simular(datos, pol, tasa)
        curvas[nombre] = curva
        filas.append({"politica": nombre, "tasa_costo": tasa, **res})
    tabla = pd.DataFrame(filas)

    # Sensibilidad al costo: la misma politica de alternar con costos distintos.
    sens = []
    for bps in [0, 1, 5, 10, 25]:
        _, res = simular(datos, politicas["Alterna 0 % / 100 % cada hora"], bps / 10_000)
        sens.append({"costo_bps": bps, **res})
    sens = pd.DataFrame(sens)

# AJUSTE RECOMENDADO
    # salida = RAIZ / "resultados"
    salida = RAIZ / "resultados_corregidos"
    salida.mkdir(parents=True, exist_ok=True)
# FIN AJUSTE RECOMENDADO    
    tabla.to_csv(salida / "entorno_politicas_fijas_validacion.csv", index=False)
    sens.to_csv(salida / "entorno_sensibilidad_costos_validacion.csv", index=False)

    fig, ax = plt.subplots(figsize=(10, 4))
    for nombre, curva in curvas.items():
        ax.plot(datos.index[: len(curva)], curva, lw=1, label=nombre)
    ax.set_title("Capital con políticas fijas (validación, costo de 10 pb)")
    ax.set_ylabel("Capital ($)")
    ax.grid(alpha=0.2)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(salida / "entorno_curvas_politicas_fijas.png", dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 3.8))
    ax.bar(sens["costo_bps"].astype(str), sens["costos_totales"])
# AJUSTE RECOMENDADO    
    # ax.set_xlabel("Costo por unidad de rotación (pb)")
    ax.set_xlabel("Comision sobre el importe negociado (pb)")
# FIN AJUSTE RECOMENDADO    
    ax.set_ylabel("Costos totales pagados ($)")
    ax.set_title("Costos al alternar 0 % / 100 % cada hora")
    ax.grid(alpha=0.2, axis="y")
    fig.tight_layout()
    fig.savefig(salida / "entorno_sensibilidad_costos.png", dpi=140)
    plt.close(fig)

    pd.set_option("display.width", 160)
    print(tabla.to_string(index=False))
    print(sens.to_string(index=False))


if __name__ == "__main__":
    main()
