"""Genera particiones, resumen y figuras a partir de un archivo .pkl existente."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from features import preparar, indicadores, cortes_temporales, ajustar_escalado, escalar, FEATURES

p = argparse.ArgumentParser()
p.add_argument('--archivo', required=True, help='Ruta al .pkl descargado')
a = p.parse_args()
out = Path('resultados'); out.mkdir(exist_ok=True)
d = preparar(pd.read_pickle(a.archivo))
x = indicadores(d)
train, val, test = cortes_temporales(x)
media, desv = ajustar_escalado(train)
for nombre, tramo in [('train',train),('validacion',val),('heldout',test)]:
    escalar(tramo,media,desv).to_csv(out/f'{nombre}.csv', index_label='timestamp')
resumen = {'fuente':a.archivo, 'filas_originales':len(d), 'filas_utiles':len(x),
           'columnas_estado':FEATURES+['posicion_actual'],
           'particiones':{n:{'filas':len(v),'inicio':str(v.index.min()),'fin':str(v.index.max())}
                          for n,v in [('train',train),('validacion',val),('heldout',test)]},
           'media_entrenamiento':media.to_dict(),'desviacion_entrenamiento':desv.to_dict(),
           'observacion':'posicion_actual la agrega el entorno; no es una columna de mercado'}
(out/'resumen.json').write_text(json.dumps(resumen,indent=2,ensure_ascii=False),encoding='utf-8')
for columna, titulo, archivo in [('close','Precio de cierre','precio.png'),('retorno_1h','Retorno por hora','retornos.png'),('volatilidad_24h','Volatilidad de 24 horas','volatilidad.png'),('volume','Volumen','volumen.png')]:
    fig,ax=plt.subplots(figsize=(10,3.5)); x[columna].plot(ax=ax,lw=.7)
    ax.set_title(titulo); ax.grid(alpha=.2); fig.tight_layout();fig.savefig(out/archivo,dpi=140);plt.close(fig)
fig,ax=plt.subplots(figsize=(7,4));x.retorno_1h.hist(ax=ax,bins=70);ax.set_title('Distribucion de retornos');fig.tight_layout();fig.savefig(out/'distribucion.png',dpi=140);plt.close(fig)
print(json.dumps(resumen,ensure_ascii=False,indent=2))
