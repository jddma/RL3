from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from gymnasium.utils.env_checker import check_env

from environment import ACCIONES, EntornoInversion
from features import FEATURES

RAIZ = Path(__file__).resolve().parents[1]


def datos_sinteticos(n=60, precio=None, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")
    close = np.full(n, 100.0) if precio == "constante" else 100 * np.cumprod(1 + rng.normal(0, 0.01, n))
    d = pd.DataFrame({"close": close}, index=idx)
    for c in FEATURES:
        d[c] = rng.normal(size=n)
    return d


def correr(env, acciones):
    env.reset(seed=0)
    ultimo = None
    for a in acciones:
        ultimo = env.step(a)
        if ultimo[2] or ultimo[3]:
            break
    return ultimo


def test_conserva_capital_sin_movimiento_ni_costos():
    env = EntornoInversion(datos_sinteticos(precio="constante"), tasa_costo=0.0)
    *_, info = correr(env, [2, 1, 0, 2, 1] * 20)
    assert info["capital"] == pytest.approx(100_000.0, abs=1e-6)


def test_efectivo_no_cambia_con_precio_variable():
    env = EntornoInversion(datos_sinteticos(), tasa_costo=0.001)
    *_, info = correr(env, [0] * 100)
    assert info["capital"] == 100_000.0
    assert info["costo_total"] == 0.0

# AJUSTE RECOMENDADO: la recompensa de t no debe usar close_{t+1} para calcular el costo
#def test_cobra_costo_al_cambiar_posicion():
#    env = EntornoInversion(datos_sinteticos(precio="constante"), tasa_costo=0.001)
#    env.reset(seed=0)
#    _, _, _, _, info = env.step(2)  # de 0 % a 100 %: rotacion 1
#    assert info["costo"] == pytest.approx(100_000 * 0.001)
#    _, _, _, _, info = env.step(2)  # mismo peso: sin costo
#    assert info["costo"] == 0.0
#    _, _, _, _, info = env.step(1)  # de 100 % a 50 %: rotacion 0.5
#    assert info["costo"] == pytest.approx(0.001 * 0.5 * (100_000 - 100))
#    assert info["operaciones"] == 2

def test_cobra_costo_al_cambiar_posicion():
    env = EntornoInversion(
        datos_sinteticos(precio="constante"),
        tasa_costo=0.001,
    )
    env.reset(seed=0)

    # Comprar usando todo el capital, incluida la comisión.
    _, _, _, _, info = env.step(2)
    compra = 100_000 / 1.001
    assert info["costo"] == pytest.approx(compra * 0.001)
    assert info["capital"] == pytest.approx(compra)
    assert info["posicion"] == pytest.approx(1.0)

    # Con precio constante, mantener 100 % no requiere operar.
    _, _, _, _, info = env.step(2)
    assert info["costo"] == 0.0

    # Vender hasta quedar con 50 % después de pagar la comisión.
    capital_antes = info["capital"]
    venta = (0.5 * capital_antes) / (1 - 0.001 * 0.5)

    _, _, _, _, info = env.step(1)
    assert info["costo"] == pytest.approx(venta * 0.001)
    assert info["capital"] == pytest.approx(
        capital_antes - venta * 0.001
    )
    assert info["posicion"] == pytest.approx(0.5)
    assert info["operaciones"] == 2

# FIN DE AJUSTE RECOMENDADO

def test_comprar_y_mantener_sin_costos_replica_el_activo():
    d = datos_sinteticos()
    env = EntornoInversion(d, tasa_costo=0.0)
    *_, info = correr(env, [2] * (len(d) - 1))
    esperado = 100_000 * d.close.iloc[-1] / d.close.iloc[0]
    assert info["capital"] == pytest.approx(esperado, rel=1e-9)


# AJUSTE RECOMENDADO: la recompensa de t no debe usar close_{t+1} para calcular el costo
#def test_recompensa_simple_y_log_incluyen_costo_una_sola_vez():
#    d = datos_sinteticos(precio="constante")
#    for nombre in ("simple", "log"):
#        env = EntornoInversion(d, tasa_costo=0.001, recompensa=nombre)
#        env.reset(seed=0)
#        _, r, _, _, info = env.step(2)
#        # precio constante: la unica perdida es el costo de entrar (10 pb)
#        esperado = -0.001 if nombre == "simple" else np.log(1 - 0.001)
#        assert r == pytest.approx(esperado) 
#        assert info["capital"] == pytest.approx(100_000 * 0.999)

def test_recompensa_simple_y_log_incluyen_costo_una_sola_vez():
    d = datos_sinteticos(precio="constante")

    for nombre in ("simple", "log"):
        env = EntornoInversion(
            d, tasa_costo=0.001, recompensa=nombre
        )
        env.reset(seed=0)
        _, r, _, _, info = env.step(2)

        capital_esperado = 100_000 / 1.001
        factor = capital_esperado / 100_000
        esperado = factor - 1 if nombre == "simple" else np.log(factor)

        assert r == pytest.approx(esperado)
        assert info["capital"] == pytest.approx(capital_esperado)
        assert info["costo_total"] == pytest.approx(
            100_000 - capital_esperado
        )

# FIN DE AJUSTE RECOMENDADO

def test_suma_de_log_recompensas_igual_al_log_del_capital():
    env = EntornoInversion(datos_sinteticos(), tasa_costo=0.001, recompensa="log")
    env.reset(seed=0)
    total, rng = 0.0, np.random.default_rng(1)
    while True:
        _, r, term, trunc, info = env.step(int(rng.integers(0, 3)))
        total += r
        if term or trunc:
            break
    assert total == pytest.approx(np.log(info["capital"] / 100_000))


def test_sin_lookahead_cambiar_el_futuro_no_altera_el_pasado():
    base = datos_sinteticos(n=80)
    alterado = base.copy()
    corte = 40
    alterado.iloc[corte + 1 :, alterado.columns.get_loc("close")] *= 3.0
    for c in FEATURES:
        alterado.iloc[corte + 1 :, alterado.columns.get_loc(c)] += 5.0
    acciones = np.random.default_rng(2).integers(0, 3, size=corte)
    a, b = EntornoInversion(base), EntornoInversion(alterado)
    oa, _ = a.reset(seed=0)
    ob, _ = b.reset(seed=0)
    for acc in acciones:
        np.testing.assert_array_equal(oa, ob)
        oa, ra, *_ = a.step(acc)
        ob, rb, *_ = b.step(acc)
        # la recompensa del paso t usa close_{t+1}; solo se compara hasta ese limite
        if a._t <= corte:
            assert ra == rb


def test_la_observacion_del_paso_t_no_contiene_el_precio_siguiente():
    d = datos_sinteticos()
    env = EntornoInversion(d)
    obs, _ = env.reset(seed=0)
    np.testing.assert_allclose(obs[:3], d[FEATURES].iloc[0].to_numpy(dtype=np.float32))
    obs, *_ = env.step(1)
    np.testing.assert_allclose(obs[:3], d[FEATURES].iloc[1].to_numpy(dtype=np.float32))
    
# AJUSTE RECOMENDADO: la observación del paso t no debe contener el precio siguiente    
#    assert obs[3] == pytest.approx(0.5)  # la posicion anterior entra al estado
    retorno = d.close.iloc[1] / d.close.iloc[0] - 1
    posicion_esperada = 0.5 * (1 + retorno) / (1 + 0.5 * retorno)
    assert obs[3] == pytest.approx(posicion_esperada)
# FIN DE AJUSTE RECOMENDADO

def test_recompensa_desconocida_se_rechaza():
    with pytest.raises(ValueError):
        EntornoInversion(datos_sinteticos(), recompensa="otra")


def test_dimensiones_y_tipos():
    env = EntornoInversion(datos_sinteticos())
    obs, info = env.reset(seed=0)
    assert obs.shape == env.observation_space.shape == (len(FEATURES) + 1,)
    assert obs.dtype == np.float32
    assert env.action_space.n == len(ACCIONES) == 3
    obs, r, term, trunc, info = env.step(1)
    assert obs.shape == (4,) and isinstance(r, float)
    assert isinstance(term, bool) and isinstance(trunc, bool)


def test_cumple_la_verificacion_de_gymnasium():
    check_env(EntornoInversion(datos_sinteticos()), skip_render_check=True)


def test_fin_de_episodio_y_error_si_se_sigue():
    d = datos_sinteticos(n=10)
    env = EntornoInversion(d)
    env.reset(seed=0)
    pasos = 0
    while True:
        _, _, term, trunc, _ = env.step(0)
        pasos += 1
        if term or trunc:
            break
    assert pasos == len(d) - 1 and trunc and not term
    with pytest.raises(RuntimeError):
        env.step(0)


def test_reproducible_con_semilla_y_episodios_aleatorios():
    d = datos_sinteticos(n=200)
    def inicio(seed):
        e = EntornoInversion(d, longitud_episodio=50, inicio_aleatorio=True)
        _, info = e.reset(seed=seed)
        return info["timestamp"]
    assert inicio(7) == inicio(7)
    pasos = 0
    e = EntornoInversion(d, longitud_episodio=50, inicio_aleatorio=True)
    e.reset(seed=3)
    while True:
        *_, term, trunc, _ = e.step(1)
        pasos += 1
        if term or trunc:
            break
    assert pasos == 50


@pytest.mark.parametrize("accion", [-1, 3, 0.5, True, "2"])
def test_acciones_invalidas_se_rechazan(accion):
    env = EntornoInversion(datos_sinteticos())
    env.reset(seed=0)
    with pytest.raises(ValueError):
        env.step(accion)


def test_accion_invalida_y_datos_invalidos():
    env = EntornoInversion(datos_sinteticos())
    env.reset(seed=0)
    malos = datos_sinteticos()
    malos.iloc[5, malos.columns.get_loc("close")] = -1
    with pytest.raises(ValueError):
        EntornoInversion(malos)

    # Datos no válidos: fechas repetidas e infinitos (en precio o indicadores).
    duplicados = datos_sinteticos()
    fechas = list(duplicados.index)
    fechas[5] = fechas[4]
    duplicados.index = pd.DatetimeIndex(fechas)
    with pytest.raises(ValueError):
        EntornoInversion(duplicados)

    for columna in ("close", FEATURES[0]):
        infinitos = datos_sinteticos()
        infinitos.iloc[5, infinitos.columns.get_loc(columna)] = np.inf
        with pytest.raises(ValueError):
            EntornoInversion(infinitos)

    # Parámetros financieros no finitos o fuera del rango aceptable.
    for capital in (0, -100, np.nan, np.inf, -np.inf):
        with pytest.raises(ValueError):
            EntornoInversion(datos_sinteticos(), capital_inicial=capital)
    for costo in (-0.01, 1.0, np.nan, np.inf, -np.inf):
        with pytest.raises(ValueError):
            EntornoInversion(datos_sinteticos(), tasa_costo=costo)
    for longitud in (0, 60, 100):
        with pytest.raises(ValueError):
            EntornoInversion(datos_sinteticos(), longitud_episodio=longitud)


def test_funciona_con_los_datos_reales_de_validacion():
# AJUSTE RECOMENDADO     
  #  d = pd.read_csv(RAIZ / "resultados" / "validacion.csv", index_col=0, parse_dates=True)
    d = pd.read_csv(RAIZ / "resultados_corregidos" / "validacion.csv", index_col=0, parse_dates=True)
# FIN AJUSTE RECOMENDADO    
    env = EntornoInversion(d)
    obs, _ = env.reset(seed=0)
    assert np.isfinite(obs).all()
    *_, info = correr(env, [2] * (len(d) - 1))
    assert info["capital"] > 0 and info["operaciones"] == 1

#AJUSTE RECOMENDADO: la recompensa de t no debe usar close_{t+1} para calcular el costo
def test_posicion_real_cambia_con_el_precio():
    # Sin costos: 50.000 en activo y 50.000 en efectivo.
    d = datos_sinteticos(n=3, precio="constante")
    d["close"] = [100.0, 110.0, 110.0]

    env = EntornoInversion(d, tasa_costo=0.0)
    env.reset(seed=0)
    obs, _, _, _, info = env.step(1)

    # El activo vale 55.000; el capital total es 105.000.
    assert info["capital"] == pytest.approx(105_000.0)
    assert info["posicion_objetivo"] == pytest.approx(0.5)
    assert info["posicion"] == pytest.approx(55_000 / 105_000)
    assert obs[3] == pytest.approx(55_000 / 105_000)


def test_repetir_objetivo_50_por_ciento_cobra_rebalanceo():
    d = datos_sinteticos(n=3, precio="constante")
    d["close"] = [100.0, 110.0, 110.0]

    env = EntornoInversion(d, tasa_costo=0.001)
    env.reset(seed=0)
    _, _, _, _, info = env.step(1)

    capital_antes = info["capital"]
    costo_anterior = info["costo_total"]

    # La subida deja más del 50 % invertido.
    assert info["posicion"] > 0.5

    # Volver al 50 % exige vender y pagar una comisión.
    _, _, _, _, info = env.step(1)

    assert info["importe_operacion"] < 0
    assert info["costo"] > 0
    assert info["costo"] == pytest.approx(
        0.001 * abs(info["importe_operacion"])
    )
    assert info["capital"] == pytest.approx(
        capital_antes - info["costo"]
    )
    assert info["posicion"] == pytest.approx(0.5)
    assert info["costo_total"] == pytest.approx(
        costo_anterior + info["costo"]
    )
    assert info["operaciones"] == 2

# FIN DE AJUSTE RECOMENDADO

# AJUSTE RECOMENDADO SEGUNDA PARTE: la recompensa de t no debe usar close_{t+1} para calcular el costo
def test_episodio_no_cruza_hueco_y_permite_elegir_segmento():
    d = datos_sinteticos(n=8, precio="constante")

    # Dos segmentos de cuatro observaciones separados por un hueco.
    fechas = pd.date_range(
        "2024-01-01", periods=12, freq="h", tz="UTC"
    )
    d.index = fechas[:4].append(fechas[8:])

    env = EntornoInversion(d, tasa_costo=0.0)
    _, info = env.reset(seed=0)

    assert len(env.segmentos) == 2
    assert info["segmento"] == 0
    assert info["timestamp"] == d.index[0]

    for paso in range(3):
        _, _, terminado, truncado, info = env.step(0)
        assert not terminado
        assert truncado == (paso == 2)

    assert info["timestamp"] == d.index[3]

    # No debe avanzar al segmento siguiente sin reset.
    with pytest.raises(RuntimeError):
        env.step(0)

    _, info = env.reset(options={"segmento": 1})
    assert info["segmento"] == 1
    assert info["timestamp"] == d.index[4]
    assert info["capital"] == pytest.approx(100_000.0)

    for _ in range(3):
        _, _, _, truncado, info = env.step(0)

    assert truncado
    assert info["timestamp"] == d.index[-1]


def test_inicio_aleatorio_respeta_segmentos_y_longitud():
    d = datos_sinteticos(n=12, precio="constante")
    fechas = pd.date_range(
        "2024-01-01", periods=16, freq="h", tz="UTC"
    )
    d.index = fechas[:6].append(fechas[10:])

    env = EntornoInversion(
        d,
        longitud_episodio=3,
        inicio_aleatorio=True,
    )

    for semilla in range(20):
        _, info = env.reset(seed=semilla)
        anterior = info["timestamp"]

        for paso in range(3):
            _, _, terminado, truncado, info = env.step(0)

            assert info["timestamp"] - anterior == pd.Timedelta(hours=1)
            assert not terminado
            assert truncado == (paso == 2)
            anterior = info["timestamp"]

    # Hay 12 filas, pero ningún segmento permite seis transiciones.
    with pytest.raises(ValueError):
        EntornoInversion(d, longitud_episodio=6)

# FIN AJUSTE RECOMENDADO SEGUNDA PARTE