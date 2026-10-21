import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
import pandas as pd
from features import preparar, indicadores, cortes_temporales, ajustar_escalado, escalar, FEATURES

def ejemplo(n=150):
    idx=pd.date_range('2024-01-01',periods=n,freq='h',tz='UTC')
    v=np.linspace(100,140,n)
    return pd.DataFrame({'open':v,'high':v+1,'low':v-1,'close':v,'volume':10},index=idx)

def test_sin_futuro():
    a=indicadores(preparar(ejemplo()))
    b=ejemplo();b.loc[b.index[-1],'close']=139.5
    # Cambiar el ultimo precio no altera indicadores de horas anteriores.
    c=indicadores(preparar(b))
    pd.testing.assert_frame_equal(a.iloc[:-1][FEATURES],c.iloc[:-1][FEATURES])

def test_particiones_y_escalado():
    x=indicadores(preparar(ejemplo()))
    tr,va,te=cortes_temporales(x)
    assert tr.index.max()<va.index.min()<te.index.min()
    m,s=ajustar_escalado(tr)
    assert np.allclose(escalar(tr,m,s)[FEATURES].mean(),0,atol=1e-8)
