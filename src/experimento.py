"""Comparación reproducible de DQN A vs B; nunca lee heldout.csv.
Ejecutar desde la raíz: python -m src.experimento
"""
from pathlib import Path
import json, random
import numpy as np
import pandas as pd
import torch
from torch import nn
from collections import deque
from .environment import EntornoInversion

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'resultados'
CONFIG = dict(semillas=[42, 123, 2026], pasos=30000, longitud_episodio=256,
              capital_inicial=100000., tasa_costo=0.001, gamma=0.99,
              lr=0.0005, batch_size=64, buffer_size=20000,
              inicio_aprendizaje=1000, frecuencia_target=500,
              epsilon_inicial=1., epsilon_final=0.05, epsilon_decay=20000)

class QNet(nn.Module):
    def __init__(self, obs, acciones):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(obs,64), nn.ReLU(), nn.Linear(64,64), nn.ReLU(), nn.Linear(64,acciones))
    def forward(self,x): return self.net(x)

def entrenar(datos, recompensa, seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    env = EntornoInversion(datos, recompensa=recompensa, tasa_costo=CONFIG['tasa_costo'],
                          longitud_episodio=CONFIG['longitud_episodio'], inicio_aleatorio=True)
    env.action_space.seed(seed)
    q = QNet(env.observation_space.shape[0], env.action_space.n)
    target = QNet(env.observation_space.shape[0], env.action_space.n)
    target.load_state_dict(q.state_dict()); optim = torch.optim.Adam(q.parameters(),lr=CONFIG['lr'])
    replay = deque(maxlen=CONFIG['buffer_size']); obs,_=env.reset(seed=seed)
    historial=[]; recompensa_episodio=0.; episodios=0
    for paso in range(CONFIG['pasos']):
        eps = CONFIG['epsilon_final'] + (CONFIG['epsilon_inicial']-CONFIG['epsilon_final'])*max(0,1-paso/CONFIG['epsilon_decay'])
        if random.random()<eps: accion=env.action_space.sample()
        else:
            with torch.no_grad(): accion=int(q(torch.as_tensor(obs,dtype=torch.float32)).argmax())
        nxt,r,terminado,truncado,info=env.step(accion)
        # Los episodios cortos terminan por límite temporal; no hacer bootstrap a través del límite.
        done=terminado or truncado
        replay.append((obs.copy(),accion,r,nxt.copy(),float(done)))
        obs=nxt; recompensa_episodio+=r
        if len(replay)>=CONFIG['inicio_aprendizaje']:
            batch=random.sample(replay,CONFIG['batch_size'])
            s=torch.as_tensor(np.array([x[0] for x in batch]),dtype=torch.float32)
            a=torch.as_tensor([x[1] for x in batch],dtype=torch.long)
            rew=torch.as_tensor([x[2] for x in batch],dtype=torch.float32)
            ns=torch.as_tensor(np.array([x[3] for x in batch]),dtype=torch.float32)
            d=torch.as_tensor([x[4] for x in batch],dtype=torch.float32)
            with torch.no_grad(): y=rew+CONFIG['gamma']*(1-d)*target(ns).max(dim=1).values
            loss=nn.functional.smooth_l1_loss(q(s).gather(1,a[:,None]).squeeze(1),y)
            optim.zero_grad();loss.backward();nn.utils.clip_grad_norm_(q.parameters(),10.);optim.step()
        if (paso+1)%CONFIG['frecuencia_target']==0:target.load_state_dict(q.state_dict())
        if done:
            historial.append(dict(recompensa=recompensa,semilla=seed,episodio=episodios,
                                  paso=paso+1,recompensa_acumulada=recompensa_episodio))
            episodios+=1;recompensa_episodio=0.;obs,_=env.reset()
    return q,historial

def validar(datos, q, recompensa, seed):
    env=EntornoInversion(datos,recompensa=recompensa,tasa_costo=CONFIG['tasa_costo'])
    obs,_=env.reset(seed=seed); curva=[CONFIG['capital_inicial']]; posiciones=[]; rotaciones=[]
    while True:
        with torch.no_grad(): accion=int(q(torch.as_tensor(obs,dtype=torch.float32)).argmax())
        obs,r,terminado,truncado,info=env.step(accion)
        curva.append(info['capital']); posiciones.append(info['posicion']);rotaciones.append(info['rotacion'])
        if terminado or truncado:break
    curva=np.array(curva); pos=np.array(posiciones);maximos=np.maximum.accumulate(curva)
    return dict(recompensa=recompensa,semilla=seed,retorno=float(curva[-1]/curva[0]-1),
                max_drawdown=float((curva/maximos-1).min()),operaciones=int(info['operaciones']),
                operaciones_por_1000_pasos=float(info['operaciones']/len(pos)*1000),
                rotacion_total=float(sum(rotaciones)),costos_totales=float(info['costo_total']),
                exposicion_0=float(np.mean(pos==0)),exposicion_50=float(np.mean(pos==.5)),
                exposicion_100=float(np.mean(pos==1)),capital_final=float(curva[-1]))

def main():
    OUT.mkdir(exist_ok=True)
    train=pd.read_csv(OUT/'train.csv',index_col=0,parse_dates=True)
    val=pd.read_csv(OUT/'validacion.csv',index_col=0,parse_dates=True)
    (OUT/'configuracion.json').write_text(json.dumps(CONFIG,indent=2),encoding='utf-8')
    filas=[];hist=[]
    for seed in CONFIG['semillas']:
        for reward in ['simple','log']:
            print(f'Entrenando {reward} semilla {seed}',flush=True)
            modelo,h=entrenar(train,reward,seed)
            hist.extend(h);filas.append(validar(val,modelo,reward,seed))
            torch.save(modelo.state_dict(),OUT/f'dqn_{reward}_{seed}.pt')
            pd.DataFrame(filas).to_csv(OUT/'comparacion_A_B.csv',index=False)
            pd.DataFrame(hist).to_csv(OUT/'historial_entrenamiento.csv',index=False)
    resumen=pd.DataFrame(filas).groupby('recompensa').agg(['mean','std'])
    resumen.to_csv(OUT/'resumen_A_B.csv')
    print(resumen.to_string())

if __name__=='__main__':main()
