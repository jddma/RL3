"""Descarga BTC/USDT horario desde Bitfinex mediante gym-trading-env."""
from datetime import datetime
from pathlib import Path
from gym_trading_env.downloader import download

def main():
    Path('data').mkdir(exist_ok=True)
    download(exchange_names=['bitfinex2'], symbols=['BTC/USDT'], timeframe='1h',
             dir='data', since=datetime(2019,1,1), until=datetime(2023,1,1))
    archivo=Path('data/bitfinex2-BTCUSDT-1h.pkl')
    if not archivo.exists():
        raise FileNotFoundError(f'No se obtuvo {archivo}')
    print('Datos descargados:', archivo)

if __name__=='__main__': main()
