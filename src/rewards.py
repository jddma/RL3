"""Contrato de recompensas: recibe capital antes y después de costos; NO descuenta costos."""
import numpy as np

def recompensa_simple(v_antes, v_despues):
    if v_antes <= 0 or v_despues <= 0: raise ValueError('Capital debe ser positivo')
    return float(v_despues / v_antes - 1)

def recompensa_log(v_antes, v_despues):
    if v_antes <= 0 or v_despues <= 0: raise ValueError('Capital debe ser positivo')
    return float(np.log(v_despues / v_antes))

RECOMPENSAS = {'simple': recompensa_simple, 'log': recompensa_log}
