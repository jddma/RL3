"""Agente DQN para el entorno de inversion.

Este modulo formaliza la logica de DQN usada para comparar las recompensas A
y B en ``experimento.py``, encapsulandola en componentes reutilizables
(``QNet``, ``ReplayBuffer``, ``DQNAgent``) para que ``train.py`` pueda
entrenar, validar y congelar un modelo final sin reescribir el ciclo de
entrenamiento cada vez.

La arquitectura, los nombres de hiperparametros y las formulas (calendario de
epsilon, perdida, sincronizacion del target) se mantienen identicos a los ya
validados en ``experimento.py`` para no introducir cambios de comportamiento
respecto a los resultados ya reportados en ``README.md``.
"""
from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass, field

import numpy as np
import torch
from torch import nn


class QNet(nn.Module):
    """Red Q totalmente conectada: obs -> 64 -> 64 -> n_acciones.

    Misma arquitectura (64-64, activaciones ReLU) usada en ``experimento.py``
    para que los checkpoints ya entrenados sigan siendo compatibles con este
    modulo.
    """

    def __init__(self, obs: int, acciones: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, acciones),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ReplayBuffer:
    """Buffer de repeticion de experiencia con muestreo uniforme.

    Guarda transiciones ``(obs, accion, recompensa, siguiente_obs, terminado)``
    en un ``deque`` de tamano maximo fijo (las transiciones mas antiguas se
    descartan primero, FIFO) y las devuelve como tensores de PyTorch listos
    para entrenar.
    """

    def __init__(self, capacidad: int = 20_000):
        if capacidad < 1:
            raise ValueError("La capacidad del buffer debe ser positiva")
        self._datos = deque(maxlen=capacidad)

    def __len__(self) -> int:
        return len(self._datos)

    def agregar(self, obs, accion: int, recompensa: float, siguiente_obs, terminado: bool) -> None:
        self._datos.append((
            np.asarray(obs, dtype=np.float32).copy(),
            int(accion),
            float(recompensa),
            np.asarray(siguiente_obs, dtype=np.float32).copy(),
            float(terminado),
        ))

    def muestrear(self, tamano_lote: int):
        """Devuelve un lote aleatorio como tensores ``(obs, accion, recompensa, siguiente_obs, terminado)``."""
        if tamano_lote > len(self._datos):
            raise ValueError("No hay suficientes transiciones para muestrear ese lote")
        lote = random.sample(self._datos, tamano_lote)
        obs = torch.as_tensor(np.array([t[0] for t in lote]), dtype=torch.float32)
        accion = torch.as_tensor([t[1] for t in lote], dtype=torch.long)
        recompensa = torch.as_tensor([t[2] for t in lote], dtype=torch.float32)
        siguiente_obs = torch.as_tensor(np.array([t[3] for t in lote]), dtype=torch.float32)
        terminado = torch.as_tensor([t[4] for t in lote], dtype=torch.float32)
        return obs, accion, recompensa, siguiente_obs, terminado


def calendario_epsilon(paso: int, inicial: float, final: float, decay: int) -> float:
    """Calendario lineal de exploracion epsilon-greedy.

    ``epsilon`` decae linealmente de ``inicial`` a ``final`` en ``decay`` pasos
    y luego se mantiene en ``final``. Misma formula usada en
    ``experimento.py`` para no alterar la dinamica de exploracion ya validada.
    """
    if decay <= 0:
        raise ValueError("decay debe ser positivo")
    avance = max(0.0, 1.0 - paso / decay)
    return final + (inicial - final) * avance


@dataclass
class ConfiguracionDQN:
    """Hiperparametros del agente DQN (mismos valores ya congelados en ``experimento.py``)."""

    gamma: float = 0.99
    lr: float = 0.0005
    batch_size: int = 64
    buffer_size: int = 20_000
    inicio_aprendizaje: int = 1_000
    frecuencia_target: int = 500
    epsilon_inicial: float = 1.0
    epsilon_final: float = 0.05
    epsilon_decay: int = 20_000
    clip_grad_norm: float = 10.0
    semilla: int | None = None
    extra: dict = field(default_factory=dict)


class DQNAgent:
    """Agente DQN con red online/target, exploracion epsilon-greedy y replay buffer.

    Encapsula el ciclo ``act -> observar -> aprender -> sincronizar_target``
    usado por ``train.py``. La politica ``greedy`` (sin exploracion) es la que
    se usa para validar sobre ``validacion.csv`` y, mas adelante, sobre el
    held-out (fuera del alcance de este entregable).
    """

    def __init__(self, obs_dim: int, n_acciones: int, config: ConfiguracionDQN | None = None):
        self.config = config or ConfiguracionDQN()
        if self.config.semilla is not None:
            torch.manual_seed(self.config.semilla)
        self.obs_dim = obs_dim
        self.n_acciones = n_acciones
        self.red = QNet(obs_dim, n_acciones)
        self.red_objetivo = QNet(obs_dim, n_acciones)
        self.red_objetivo.load_state_dict(self.red.state_dict())
        self.red_objetivo.eval()
        self.optimizador = torch.optim.Adam(self.red.parameters(), lr=self.config.lr)
        self.buffer = ReplayBuffer(self.config.buffer_size)
        self._pasos_aprendidos = 0

    # ------------------------------------------------------------------
    def epsilon(self, paso: int) -> float:
        return calendario_epsilon(
            paso, self.config.epsilon_inicial, self.config.epsilon_final, self.config.epsilon_decay
        )

    def actuar(self, obs, paso: int, explorar: bool = True) -> int:
        """Elige una accion epsilon-greedy (o puramente greedy si ``explorar=False``)."""
        if explorar and random.random() < self.epsilon(paso):
            return random.randrange(self.n_acciones)
        return self.actuar_greedy(obs)

    def actuar_greedy(self, obs) -> int:
        """Politica greedy (sin exploracion); usada en validacion/evaluacion."""
        with torch.no_grad():
            valores = self.red(torch.as_tensor(obs, dtype=torch.float32))
            return int(valores.argmax().item())

    def observar(self, obs, accion: int, recompensa: float, siguiente_obs, terminado: bool) -> None:
        self.buffer.agregar(obs, accion, recompensa, siguiente_obs, terminado)

    def aprender(self) -> float | None:
        """Un paso de descenso de gradiente sobre un lote del replay buffer.

        Devuelve la perdida (``float``) o ``None`` si todavia no hay
        suficientes transiciones (``inicio_aprendizaje``) para entrenar.
        """
        if len(self.buffer) < self.config.inicio_aprendizaje:
            return None
        obs, accion, recompensa, siguiente_obs, terminado = self.buffer.muestrear(self.config.batch_size)
        with torch.no_grad():
            objetivo = recompensa + self.config.gamma * (1 - terminado) * self.red_objetivo(siguiente_obs).max(dim=1).values
        estimado = self.red(obs).gather(1, accion[:, None]).squeeze(1)
        perdida = nn.functional.smooth_l1_loss(estimado, objetivo)
        self.optimizador.zero_grad()
        perdida.backward()
        nn.utils.clip_grad_norm_(self.red.parameters(), self.config.clip_grad_norm)
        self.optimizador.step()
        self._pasos_aprendidos += 1
        if self._pasos_aprendidos % self.config.frecuencia_target == 0:
            self.sincronizar_target()
        return float(perdida.item())

    def sincronizar_target(self) -> None:
        """Copia los pesos de la red online hacia la red objetivo (target network)."""
        self.red_objetivo.load_state_dict(self.red.state_dict())

    # ------------------------------------------------------------------
    def guardar(self, ruta) -> None:
        """Guarda los pesos de la red online (compatible con ``torch.load``)."""
        torch.save(self.red.state_dict(), ruta)

    def cargar(self, ruta, mapa_dispositivo: str = "cpu") -> None:
        """Carga pesos en la red online y sincroniza la red objetivo."""
        estado = torch.load(ruta, map_location=mapa_dispositivo)
        self.red.load_state_dict(estado)
        self.sincronizar_target()
