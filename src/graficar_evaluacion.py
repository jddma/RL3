"""Genera figuras a partir de las trayectorias del held-out."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ENTRADA = ROOT / "evaluacion_heldout"
SALIDA = ENTRADA / "figuras"

ESTRATEGIAS = {
    "DQN": ("DQN", "tab:blue"),
    "Comprar_y_mantener": ("Comprar y mantener", "tab:orange"),
    "Aleatorio": ("Aleatorio", "tab:gray"),
}


def main():
    SALIDA.mkdir(exist_ok=True)
    metricas = pd.read_csv(ENTRADA / "metricas.csv")

    for segmento in sorted(metricas["segmento"].unique()):
        trayectorias = {}
        for nombre in ESTRATEGIAS:
            trayectorias[nombre] = pd.read_csv(
                ENTRADA / f"trayectoria_{nombre}_segmento_{segmento}.csv",
                parse_dates=["timestamp"],
            )

        # Capital y drawdown: comparación bajo las mismas condiciones.
        fig, ejes = plt.subplots(
            2, 1, figsize=(11, 7), sharex=True
        )

        for nombre, (etiqueta, color) in ESTRATEGIAS.items():
            datos = trayectorias[nombre]
            ejes[0].plot(
                datos["timestamp"], datos["capital"],
                label=etiqueta, color=color, linewidth=1,
            )
            ejes[1].plot(
                datos["timestamp"], datos["drawdown"] * 100,
                label=etiqueta, color=color, linewidth=1,
            )

        ejes[0].set_title(
            f"Held-out — segmento {segmento}: capital y drawdown"
        )
        ejes[0].set_ylabel("Capital (unidades monetarias)")
        ejes[0].legend()
        ejes[1].set_ylabel("Drawdown (%)")
        ejes[1].set_xlabel("Fecha (UTC)")

        for eje in ejes:
            eje.grid(alpha=0.25)

        fig.autofmt_xdate()
        fig.tight_layout()
        fig.savefig(
            SALIDA / f"capital_drawdown_segmento_{segmento}.png",
            dpi=160,
        )
        plt.close(fig)

        # Distinguir la acción objetivo de la posición real.
        dqn = trayectorias["DQN"].iloc[1:].copy()
        fig, ejes = plt.subplots(
            2, 1, figsize=(11, 6), sharex=True
        )

        ejes[0].step(
            dqn["timestamp"],
            dqn["posicion_objetivo"] * 100,
            where="pre",
            linewidth=0.8,
            color="tab:blue",
        )
        ejes[0].set_ylabel("Objetivo elegido (%)")
        ejes[0].set_yticks([0, 50, 100])
        ejes[0].set_title(
            f"DQN — segmento {segmento}: decisiones y exposición"
        )

        ejes[1].plot(
            dqn["timestamp"],
            dqn["posicion"] * 100,
            linewidth=0.8,
            color="tab:green",
        )
        ejes[1].set_ylabel("Exposición al final del paso (%)")
        ejes[1].set_xlabel("Fecha del resultado, t+1 (UTC)")

        for eje in ejes:
            eje.set_ylim(-5, 105)
            eje.grid(alpha=0.25)

        fig.autofmt_xdate()
        fig.tight_layout()
        fig.savefig(
            SALIDA / f"acciones_posiciones_DQN_segmento_{segmento}.png",
            dpi=160,
        )
        plt.close(fig)

        # Costos acumulados.
        fig, eje = plt.subplots(figsize=(11, 4))
        for nombre, (etiqueta, color) in ESTRATEGIAS.items():
            datos = trayectorias[nombre]
            eje.plot(
                datos["timestamp"],
                datos["costo"].cumsum(),
                label=etiqueta,
                color=color,
                linewidth=1,
            )

        eje.set_title(
            f"Held-out — segmento {segmento}: costos acumulados"
        )
        eje.set_ylabel("Costos (unidades monetarias)")
        eje.set_xlabel("Fecha (UTC)")
        eje.grid(alpha=0.25)
        eje.legend()
        fig.autofmt_xdate()
        fig.tight_layout()
        fig.savefig(
            SALIDA / f"costos_segmento_{segmento}.png",
            dpi=160,
        )
        plt.close(fig)

    print("Figuras generadas:")
    for archivo in sorted(SALIDA.glob("*.png")):
        print(archivo.name)


if __name__ == "__main__":
    main()