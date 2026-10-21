"""Estrategias de referencia para el entorno de inversion."""

import numpy as np

class ComprarYMantener:
    """Objetivo 100 % durante todo el episodio."""

    def actuar_greedy(self, observacion):
        return 2

class AgenteAleatorio:
    """Seleccion uniforme entre las tres acciones, con semilla fija."""

    def __init__(self, semilla=42):
        self.reiniciar(semilla)

    def reiniciar(self, semilla):
        self._rng = np.random.default_rng(semilla)

    def actuar_greedy(self, observacion):
        # Se conserva el nombre del metodo para compartir la interfaz
        # con DQN; esta politica sigue siendo aleatoria.
        return int(self._rng.integers(0, 3))