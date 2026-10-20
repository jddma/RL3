import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from src.agent import ConfiguracionDQN, DQNAgent, QNet, ReplayBuffer, calendario_epsilon


class TestQNet(unittest.TestCase):
    def test_forma_salida(self):
        red = QNet(obs=5, acciones=3)
        lote = torch.randn(8, 5)
        salida = red(lote)
        self.assertEqual(tuple(salida.shape), (8, 3))


class TestCalendarioEpsilon(unittest.TestCase):
    def test_limites(self):
        self.assertAlmostEqual(calendario_epsilon(0, 1.0, 0.05, 1000), 1.0)
        self.assertAlmostEqual(calendario_epsilon(1000, 1.0, 0.05, 1000), 0.05)
        self.assertAlmostEqual(calendario_epsilon(5000, 1.0, 0.05, 1000), 0.05)

    def test_monotono_decreciente(self):
        valores = [calendario_epsilon(p, 1.0, 0.05, 1000) for p in range(0, 1000, 100)]
        self.assertEqual(valores, sorted(valores, reverse=True))

    def test_decay_invalido(self):
        with self.assertRaises(ValueError):
            calendario_epsilon(10, 1.0, 0.05, 0)


class TestReplayBuffer(unittest.TestCase):
    def _transicion(self, i):
        obs = np.full(4, float(i), dtype=np.float32)
        return obs, i % 3, float(i), obs + 1, False

    def test_agregar_y_muestrear(self):
        buffer = ReplayBuffer(capacidad=50)
        for i in range(20):
            buffer.agregar(*self._transicion(i))
        self.assertEqual(len(buffer), 20)
        obs, accion, recompensa, siguiente_obs, terminado = buffer.muestrear(8)
        self.assertEqual(tuple(obs.shape), (8, 4))
        self.assertEqual(tuple(accion.shape), (8,))
        self.assertEqual(tuple(recompensa.shape), (8,))
        self.assertEqual(tuple(siguiente_obs.shape), (8, 4))
        self.assertEqual(tuple(terminado.shape), (8,))

    def test_capacidad_fifo(self):
        buffer = ReplayBuffer(capacidad=10)
        for i in range(25):
            buffer.agregar(*self._transicion(i))
        self.assertEqual(len(buffer), 10)

    def test_muestreo_insuficiente(self):
        buffer = ReplayBuffer(capacidad=10)
        buffer.agregar(*self._transicion(0))
        with self.assertRaises(ValueError):
            buffer.muestrear(5)

    def test_capacidad_invalida(self):
        with self.assertRaises(ValueError):
            ReplayBuffer(capacidad=0)


class TestDQNAgent(unittest.TestCase):
    def _config(self, semilla=42, inicio_aprendizaje=5, frecuencia_target=3):
        return ConfiguracionDQN(
            buffer_size=100, batch_size=4, inicio_aprendizaje=inicio_aprendizaje,
            frecuencia_target=frecuencia_target, semilla=semilla,
        )

    def test_reproducibilidad_con_semilla(self):
        agente_a = DQNAgent(obs_dim=4, n_acciones=3, config=self._config(semilla=123))
        agente_b = DQNAgent(obs_dim=4, n_acciones=3, config=self._config(semilla=123))
        obs = np.zeros(4, dtype=np.float32)
        self.assertEqual(agente_a.actuar_greedy(obs), agente_b.actuar_greedy(obs))

    def test_aprender_none_antes_de_inicio(self):
        agente = DQNAgent(obs_dim=4, n_acciones=3, config=self._config(inicio_aprendizaje=10))
        obs = np.zeros(4, dtype=np.float32)
        for _ in range(5):
            agente.observar(obs, 0, 0.0, obs, False)
            self.assertIsNone(agente.aprender())

    def test_aprender_devuelve_perdida_tras_inicio(self):
        agente = DQNAgent(obs_dim=4, n_acciones=3, config=self._config(inicio_aprendizaje=5))
        obs = np.zeros(4, dtype=np.float32)
        perdida = None
        for i in range(10):
            agente.observar(obs, i % 3, 0.1, obs, False)
            perdida = agente.aprender()
        self.assertIsNotNone(perdida)
        self.assertIsInstance(perdida, float)

    def test_guardar_y_cargar(self):
        agente = DQNAgent(obs_dim=4, n_acciones=3, config=self._config())
        obs = np.zeros(4, dtype=np.float32)
        accion_esperada = agente.actuar_greedy(obs)
        with tempfile.TemporaryDirectory() as tmp:
            ruta = Path(tmp) / "modelo.pt"
            agente.guardar(ruta)
            otro = DQNAgent(obs_dim=4, n_acciones=3, config=self._config(semilla=999))
            otro.cargar(ruta)
            self.assertEqual(otro.actuar_greedy(obs), accion_esperada)


if __name__ == "__main__":
    unittest.main()
