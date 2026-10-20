"""Descarga (si hace falta) y genera resultados."""
from pathlib import Path
import subprocess,sys

def ejecutar(*args): subprocess.run([sys.executable,*args],check=True)

def main():
    archivo=Path('data/bitfinex2-BTCUSDT-1h.pkl')
    if not archivo.exists(): ejecutar('src/descargar_datos.py')
    else: print('Reutilizando:',archivo)
    ejecutar('src/analizar_datos.py','--archivo',str(archivo))
    print('Resultados disponibles en resultados/')

if __name__=='__main__':main()
