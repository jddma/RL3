"""Preparacion causal de precios y variables para el agente."""
import numpy as np
import pandas as pd

FEATURES = ['retorno_1h', 'distancia_media_24h', 'volatilidad_24h']

def preparar(df):
    d = df.copy()
    d.columns = [str(c).lower().strip() for c in d.columns]
    if 'date' in d.columns and 'timestamp' not in d.columns:
        d = d.rename(columns={'date': 'timestamp'})
    if 'timestamp' in d.columns:
        d['timestamp'] = pd.to_datetime(d['timestamp'], utc=True)
        d = d.set_index('timestamp')
    if not isinstance(d.index, pd.DatetimeIndex):
        raise ValueError('Se necesita indice de fechas o columna timestamp/date')
    d.index = pd.to_datetime(d.index, utc=True)
    d = d.sort_index()
    if d.index.has_duplicates:
        raise ValueError('Hay fechas duplicadas; revisar fuente antes de continuar')
    requeridas = ['open', 'high', 'low', 'close', 'volume']
    if not set(requeridas).issubset(d.columns):
        raise ValueError(f'Faltan columnas: {set(requeridas)-set(d.columns)}')
    d[requeridas] = d[requeridas].apply(pd.to_numeric, errors='coerce')
    if d[requeridas].isna().any().any():
        raise ValueError('Hay valores vacios o no numericos; no se rellenan con datos futuros')
    if (d[['open','high','low','close']] <= 0).any().any() or (d.volume < 0).any():
        raise ValueError('Precios no positivos o volumen negativo')
    if ((d.high < d[['open','close','low']].max(axis=1)) | (d.low > d[['open','close','high']].min(axis=1))).any():
        raise ValueError('OHLC incoherente')
    return d[requeridas]

def indicadores(d):
    x = d.copy()
    x['retorno_1h'] = x.close.pct_change(fill_method=None)
    media = x.close.rolling(24, min_periods=24).mean()
    x['distancia_media_24h'] = x.close / media - 1
    x['volatilidad_24h'] = x['retorno_1h'].rolling(24, min_periods=24).std()
    return x.dropna(subset=FEATURES)

def cortes_temporales(d, train=0.7, validation=0.15):
    if not 0 < train < 1 or not 0 < validation < 1 or train + validation >= 1:
        raise ValueError('Proporciones invalidas')
    a, b = int(len(d)*train), int(len(d)*(train+validation))
    if a < 2 or b <= a or len(d)-b < 2:
        raise ValueError('Datos insuficientes')
    return d.iloc[:a].copy(), d.iloc[a:b].copy(), d.iloc[b:].copy()

def ajustar_escalado(train):
    media = train[FEATURES].mean()
    desviacion = train[FEATURES].std().replace(0, 1)
    return media, desviacion

def escalar(d, media, desviacion):
    x = d.copy()
    x[FEATURES] = (x[FEATURES]-media)/desviacion
    return x
