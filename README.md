# Agente de inversión con Reinforcement Learning

Entorno de simulación de inversión en Bitcoin (BTC/USDT, datos horarios),
comparación de dos funciones de recompensa y entrenamiento de un agente DQN
que decide qué fracción del capital (0 %, 50 % o 100 %) mantener invertida,
con costos de transacción y evaluación sin fuga de información hacia el
futuro (*lookahead-safe*).

## Objetivo

1. Comparar retorno simple neto (A) y log-retorno neto (B) como función de
   recompensa, usando el mismo agente DQN, los mismos datos y el mismo costo
   de operación, sin consultar el conjunto held-out.
2. Entrenar y congelar un agente DQN final con la recompensa elegida,
   documentando arquitectura, hiperparámetros, semillas y criterio de
   selección del modelo, listo para una evaluación posterior en datos no
   vistos (held-out).

## Instalación y ejecución rápida

Desde esta carpeta:

```bash
pip install -r requirements.txt
python -m unittest discover -s tests -v
python -m src.experimento   # compara las recompensas A y B (6 entrenamientos de 30.000 pasos)
python -m src.train         # entrena y congela el agente DQN final (recompensa A)
```

**Atención:** `python -m src.experimento` ejecuta 6 entrenamientos de 30.000
pasos y puede tardar varios minutos o más según el equipo; el archivo
`comparacion_A_B.csv` se genera progresivamente. Entrenar sobrescribe los CSV
y checkpoints existentes; conservar una copia de los resultados originales si
se quieren preservar. El held-out (`heldout.csv`) no se usa en ninguno de los
dos comandos anteriores: se reserva para una evaluación final posterior.

## Entorno y contrato de recompensas

La función de recompensa recibe `(capital_antes, capital_despues)`; el
entorno (`environment.py`) descuenta una sola vez
`tasa_costo * abs(posicion_nueva - posicion_anterior) * capital_antes` al
calcular el capital siguiente. Las funciones de recompensa (`rewards.py`) no
vuelven a modificar el capital ni a descontar el costo nuevamente.
`environment.py` implementa `recompensa='simple'|'log'`, y `rewards.py`
documenta y permite probar ambas fórmulas de forma independiente.

## Diseño experimental (comparación de recompensas)

- Datos de entrenamiento: `resultados/train.csv`; validación:
  `resultados/validacion.csv`; held-out (`resultados/heldout.csv`) reservado
  para una evaluación final posterior.
- DQN: red 64-64 ReLU, Adam, replay buffer, target network, exploración
  epsilon-greedy.
- 3 semillas emparejadas: 42, 123, 2026; 30.000 pasos por semilla y
  recompensa.
- Episodios de entrenamiento de 256 pasos con inicio aleatorio. Validación
  cronológica completa y política greedy.
- Costos: 0,001 por unidad de cambio de posición; capital inicial 100.000.
- La métrica de selección es el retorno neto medio en validación, condicionada
  a drawdown y estabilidad; se evita seleccionar por una sola semilla.

### Evidencias de la comparación A vs B
- `resultados/configuracion.json`: parámetros.
- `resultados/historial_entrenamiento.csv`: recompensa por episodio.
- `resultados/comparacion_A_B.csv`: resultados por semilla y función.
- `resultados/resumen_A_B.csv`: medias y desviaciones estándar.

### Análisis de conducta y reward hacking
1. **Inactividad:** `exposicion_0` próxima a 1 y `operaciones` próximas a 0. Comparar contra política siempre efectivo (retorno 0).
2. **Sobreoperación:** `operaciones_por_1000_pasos` y `rotacion_total` altos; revisar si los costos absorben retornos.
3. **Riesgo excesivo:** `exposicion_100` alta y drawdown profundo; comparar con exposición fija 100 %.
4. **Explotación de simulación:** verificar que el agente no observe el precio t+1 y que los costos se descuenten exactamente una vez. El entorno supone ejecución al cierre sin spread/slippage y una posición objetivo constante entre decisiones; estas simplificaciones limitan la validez externa.
5. **Estabilidad:** comparar la desviación estándar entre semillas, no solamente el mejor resultado.

### Criterio para recomendar una recompensa
Elegir B si mejora de forma consistente retorno neto o drawdown sin elevar
mucho la rotación, ni inducir inactividad. Elegir A si B no muestra ventajas
consistentes. Si ninguna supera efectivo o los baselines fijos con riesgo
aceptable, declarar el resultado inconcluso y no usar held-out para ajustar.
Con los datos disponibles, la recompensa elegida para el agente final es
**A (retorno simple neto)**, por menor costo y rotación promedio, menor
dispersión entre semillas y retorno medio apenas superior; no se demuestra
superioridad estadística firme (solo tres semillas, pérdidas severas en
ambas variantes y diferencia de retorno media pequeña).

### Nota técnica
Las funciones de recompensa A y B son transformaciones crecientes del mismo
cociente de capital por paso, pero su suma acumulada puede inducir políticas
diferentes. El log-retorno acumulado equivale al logaritmo del crecimiento
total del capital para trayectorias positivas; no garantiza por sí mismo un
menor drawdown. El término `truncated` del entorno indica finalización
temporal y aquí se trata como final del episodio de entrenamiento.

## Agente DQN y entrenamiento

El agente DQN se implementa en módulos reutilizables y documentados:

- `src/agent.py`: `QNet`, `ReplayBuffer`, calendario de epsilon y la clase
  `DQNAgent` (red online/target, política epsilon-greedy, aprendizaje con
  *smooth L1 loss* y recorte de gradiente, sincronización periódica del
  target, guardado/carga de checkpoints).
- `src/train.py`: script/CLI de entrenamiento, validación y congelamiento del
  modelo final, con semillas fijas y registro de historial.
- `tests/test_agente.py`: pruebas unitarias de `QNet`, el calendario de
  epsilon, el `ReplayBuffer` y la reproducibilidad/guardado del `DQNAgent`.

### Arquitectura y algoritmo

- **Algoritmo:** DQN (Deep Q-Network) con red objetivo (*target network*) y
  repetición de experiencia (*experience replay*).
- **Red:** `obs -> Linear(64) -> ReLU -> Linear(64) -> ReLU -> Linear(n_acciones=3)`.
- **Optimizador:** Adam, `lr = 0.0005`.
- **Descuento:** `gamma = 0.99`.
- **Lote de entrenamiento:** `batch_size = 64`.
- **Replay buffer:** tamaño `20.000`, FIFO (`deque`), muestreo uniforme;
  aprendizaje comienza tras `inicio_aprendizaje = 1.000` transiciones.
- **Red objetivo:** se sincroniza con la red online cada
  `frecuencia_target = 500` pasos de aprendizaje.
- **Exploración:** epsilon-greedy con calendario lineal,
  `epsilon: 1.00 -> 0.05` en `epsilon_decay = 20.000` pasos, luego constante.
- **Pérdida:** Huber/Smooth L1 entre el valor Q estimado y el objetivo
  `r + gamma * (1 - terminado) * max_a Q_objetivo(s', a)`.
- **Recorte de gradiente:** norma máxima 10.
- **Pasos por corrida:** `30.000`; episodios de entrenamiento de `256` pasos
  con inicio aleatorio (no se hace *bootstrap* a través del límite del
  episodio).
- **Criterio de parada:** número fijo de pasos de entrenamiento (`30.000`);
  no hay parada temprana basada en validación para evitar ajustar sobre datos
  de evaluación.
- **Capital inicial:** `$100.000`; **costo de transacción:** `0,10%` por
  unidad de rotación.
- **Recompensa final usada para congelar el modelo:** **A (retorno simple
  neto)**.

### Semillas, versiones y reproducibilidad

- Semillas fijadas: `random`, `numpy` y `torch` vía `fijar_semillas(seed)`;
  además `env.action_space.seed(seed)` y `env.reset(seed=seed)`.
- Semillas de entrenamiento/validación: `42`, `123`, `2026`.
- Entrenamiento ejecutado en **CPU**; con las mismas semillas los resultados
  son reproducibles en CPU. **Advertencia de no determinismo:** si se corre en
  GPU, algunas operaciones de cuDNN no son deterministas salvo que se active
  `torch.use_deterministic_algorithms(True)` (no requerido aquí porque no se
  usó GPU).
- Versiones utilizadas al generar los resultados de este entregable:
  `torch 2.14.1+cpu`, `numpy 2.5.4`, `pandas 3.0.6`, `gymnasium 1.4.0`
  (Python 3.12).

### Cómo entrenar el agente final

```bash
python -m src.train                       # entrena 3 semillas, valida y congela el modelo final
python -m src.train --pasos 2000          # corrida rápida de verificación
python -m src.train --semillas 42         # entrenar solo una semilla
python -m src.train --recompensa log      # entrenar con la recompensa B
python -m src.train --no-congelar         # entrenar/validar sin seleccionar modelo final
```

### Regla de selección y modelo congelado

De las tres semillas entrenadas con la recompensa final (A), se congela la
que obtiene el **mayor retorno en validación**, siempre que su máximo
drawdown no sea peor que `1,15x` la mediana de drawdown del grupo (evita
congelar una semilla con retorno alto pero riesgo desproporcionado). El
held-out **no** se usa para esta decisión.

**Resultado de esta ejecución** (validación, `resultados/validacion_entrenamiento_agente.csv`):

| Semilla | Retorno | Max Drawdown | Operaciones | Costos totales |
| :--- | :--- | :--- | :--- | :--- |
| 42 | -61.04% | -65.75% | 999 | 30,328.97 |
| 123 | -84.17% | -84.21% | 1,947 | 68,434.65 |
| **2026 (congelada)** | **-45.34%** | **-45.76%** | 416 | 15,547.66 |

La semilla **2026** domina en retorno y en drawdown frente a las otras dos, por
lo que la regla de selección la elige de forma inequívoca. El checkpoint
congelado queda en `resultados/modelo_final.pt`, con su configuración y
métricas en `resultados/configuracion_final.json`, listo para una evaluación
posterior sobre `heldout.csv`.

**Nota honesta sobre el resultado:** el agente sigue perdiendo capital de
forma importante en validación interna (todas las semillas quedan muy por
debajo del baseline "siempre efectivo", que conserva 0 % de retorno). Esto no
se oculta ni se "arregla" artificialmente: se documenta como limitación a
discutir en el análisis final del proyecto.

### Evidencias del entrenamiento del agente
- `src/agent.py`, `src/train.py`
- `tests/test_agente.py`
- `resultados/configuracion_agente.json` (configuración de la última corrida)
- `resultados/configuracion_final.json` (modelo congelado + regla de selección)
- `resultados/historial_entrenamiento_agente.csv` (recompensa/pérdida/epsilon por episodio)
- `resultados/validacion_entrenamiento_agente.csv` (métricas por semilla)
- `resultados/dqn_simple_{42,123,2026}.pt` (checkpoints por semilla)
- `resultados/modelo_final.pt` (checkpoint congelado, semilla 2026)
