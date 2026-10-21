# Agente inversor con aprendizaje por refuerzo

Proyecto académico de Simulación y Aprendizaje por Refuerzo. Implementa
un agente DQN que decide qué proporción de su capital invertir en BTC/USDT:
0 %, 50 % o 100 %.

Integra preparación de datos, entorno de inversión, comparación de
recompensas, entrenamiento y evaluación frente a estrategias de referencia.

## 1. Componentes implementados

| Actividad | Componentes |
|---|---|
| 1. Datos y estado | Limpieza, indicadores causales, separación temporal y escalado |
| 2. Entorno | Interfaz Gymnasium, posiciones, capital, costos y segmentos continuos |
| 3. Recompensas | Comparación experimental de retorno simple y log-retorno netos |
| 4. Agente DQN | Red online/target, replay buffer, entrenamiento y modelo congelado |
| 5. Evaluación | Baselines, métricas held-out, trayectorias, figuras y análisis |

El informe general y las reflexiones individuales se consolidan como parte
de la Actividad 6.

## 2. Estructura principal

```text
Agente_Inversor/
├── README.md
├── requirements.txt
├── src/
│   ├── features.py
│   ├── environment.py
│   ├── rewards.py
│   ├── agent.py
│   ├── train.py
│   ├── experimento.py
│   ├── baselines.py
│   ├── evaluate.py
│   ├── demo_entorno.py
│   ├── eda_actividad1.py
│   └── graficar_evaluacion.py
│   ├── preparar_datos_corregidos.py
│   ├── congelar_modelo.py
├── tests/
├── resultados_corregidos/
├── eda_actividad1/
├── resultados_comparacion_recompensas/
├── modelo_congelado/
├── evaluacion_validacion/
├── evaluacion_heldout/
│   └── figuras/
└── docs/
    ├── resultados_actividad1.md
    └── resultados_actividad5.md
```

`resultados/` conserva datos y figuras anteriores a las correcciones.
No es la fuente de los resultados finales aquí reportados.

## 3. Instalación y pruebas

Entorno utilizado: Windows, PowerShell, Python 3.12.10 y PyTorch
2.14.1+cpu.

```powershell
uv venv --python 3.12.10 --prompt RL-U5
.\.venv\Scripts\Activate.ps1
uv pip install -r requirements.txt
uv pip install "torch==2.14.1" --index-url https://download.pytorch.org/whl/cpu
uv pip check
python -m pytest .\tests -q
```

Omitir la creación del entorno si ya existe.

La última ejecución reportada obtuvo 42 pruebas aprobadas y dos
advertencias de Gymnasium por los límites infinitos de los indicadores
del espacio de observación. Estas advertencias no son errores.

Los módulos de evaluación se verificaron mediante su ejecución y la
reproducción del resultado de validación del modelo congelado; no se
atribuyen a esas 42 pruebas comprobaciones que no contienen.

## 4. Datos y prevención de información futura

El análisis exploratorio actualizado y la justificación del estado se
documentan en [Resultados de la Actividad 1](docs/resultados_actividad1.md).
Las estadísticas y cinco figuras se encuentran en `eda_actividad1/`
y se generan con `python -m src.eda_actividad1`.

Fuente histórica: Bitfinex2, BTC/USDT, frecuencia horaria.

Los datos vigentes están en `resultados_corregidos/`:

| Partición | Filas | Segmentos continuos |
|---|---:|---:|
| Entrenamiento | 21.606 | 31 |
| Validación | 4.939 | 1 |
| Held-out | 4.916 | 2 |

La división original fue cronológica, aproximadamente 70/15/15.
Después de corregir los huecos se conservaron las fechas de corte,
sin recalcular esos porcentajes.

Los indicadores se calculan dentro de segmentos horarios continuos,
con 24 observaciones de calentamiento. No se imputaron precios.
El escalado se ajustó únicamente con entrenamiento.

La serie local se recuperó de las columnas OHLCV de los CSV anteriores.
No se recuperaron las 24 observaciones iniciales descartadas en el
procesamiento original.

El held-out no se utilizó para seleccionar recompensas, hiperparámetros
ni checkpoint. Después de evaluarlo no se reajustó el modelo.

Los scripts históricos `descargar_datos.py`, `analizar_datos.py` y
`ejecutar_todo.py` se conservan como antecedentes. No constituyen por sí
solos el procedimiento actualizado de regeneración de los CSV corregidos.

## 5. Estado, acciones y entorno

La observación contiene:

- Retorno de una hora, escalado.
- Distancia a la media móvil de 24 observaciones horarias, escalada.
- Volatilidad de 24 retornos horarios, escalada.
- Proporción real del capital invertido, entre 0 y 1.

| Acción | Objetivo de inversión |
|---|---:|
| 0 | 0 % |
| 1 | 50 % |
| 2 | 100 % |

En t, el agente observa información hasta el cierre y elige un objetivo.
El entorno calcula la compra o venta necesaria y cobra una comisión
del 0,1 % sobre el importe negociado.

El objetivo se define sobre el capital después de costos. Luego se
aplica el movimiento del precio de t a t+1 al activo mantenido.

La posición real cambia con el precio. Repetir el objetivo 50 % puede
requerir rebalanceo y generar costos.

Cada episodio permanece dentro de un segmento continuo. `reset()`
reinicia capital y posición; puede elegirse el segmento mediante:

```python
env.reset(options={"segmento": 0})
```

Supuestos: capital inicial de 100.000 unidades monetarias, ejecución
al cierre observado, sin spread, deslizamiento, apalancamiento ni ventas
en corto. El efectivo no genera intereses y no se fuerza liquidación final.

### Ejemplo de uso

```python
import pandas as pd
from src.environment import EntornoInversion

datos = pd.read_csv(
    "resultados_corregidos/train.csv",
    index_col=0,
    parse_dates=True,
)

env = EntornoInversion(datos, tasa_costo=0.001, recompensa="simple")
obs, info = env.reset(seed=42, options={"segmento": 0})
env.action_space.seed(42)

while True:
    accion = env.action_space.sample()
    obs, recompensa, terminado, truncado, info = env.step(accion)
    if terminado or truncado:
        break

print(info["capital"], info["costo_total"])
```

Este ejemplo recorre un episodio dentro del segmento seleccionado.

## 6. Comparación de recompensas

Las recompensas reciben el capital antes y después del paso:

- Simple: `V_siguiente / V_actual - 1`.
- Logarítmica: `log(V_siguiente / V_actual)`.

Los costos ya están descontados del capital; no se descuentan nuevamente
en la recompensa.

Se ejecutaron seis entrenamientos: ambas recompensas con semillas
42, 123 y 2026, y 30.000 pasos por corrida.

| Métrica de validación | Simple | Log |
|---|---:|---:|
| Retorno medio | −57,10 % | −57,11 % |
| Desviación del retorno | 13,59 pp | 21,52 pp |
| Máximo drawdown medio | −60,39 % | −63,29 % |
| Costos medios | 33.128,30 | 35.973,44 |

Se eligió recompensa simple por su retorno medio prácticamente igual,
menor dispersión, menor caída promedio y menores costos promedio.
No se afirma superioridad estadística con tres semillas.

```powershell
python -m src.experimento
```

Las evidencias se guardan en `resultados_comparacion_recompensas/`.
El comando requiere que esa carpeta no exista para evitar sobrescrituras.

## 7. DQN y entrenamiento

| Parámetro | Valor |
|---|---|
| Red | 4 entradas → 64 ReLU → 64 ReLU → 3 salidas |
| Optimizador | Adam |
| Learning rate | 0,0005 |
| Gamma | 0,99 |
| Batch size | 64 |
| Replay buffer | 20.000 transiciones |
| Inicio del aprendizaje | 1.000 transiciones |
| Sincronización target | Cada 500 actualizaciones de aprendizaje |
| Exploración | Epsilon de 1,00 a 0,05 en 20.000 pasos |
| Pasos por corrida | 30.000 |
| Longitud del episodio | 256 transiciones |
| Pérdida | Smooth L1 |
| Recorte de gradiente | Norma máxima 10 |

Se fijan semillas de Python, NumPy, PyTorch y entorno.
Los episodios comienzan aleatoriamente dentro de segmentos admisibles.

El límite artificial de 256 pasos permite bootstrap cuando existe
continuación dentro del segmento. Al final del segmento disponible
no se estima continuación.

### Ejemplo de corrida de verificación

Entrena durante 2.000 pasos con la semilla 42, evalúa en validación
y guarda las salidas en `resultados_dqn_prueba`, sin congelar un modelo final.

```powershell
# Prueba breve del entrenamiento y la generación de archivos
python -m src.train --pasos 2000 --semillas 42 --no-congelar --salida resultados_dqn_prueba
```

Esta corrida comprueba el funcionamiento; no se utiliza para seleccionar
la recompensa ni el modelo definitivo.

Las salidas de `train.py` se guardan por defecto en `resultados_dqn/`.
La comparación definitiva se realizó con `src.experimento`, y se congeló
un checkpoint de esa comparación sin repetir el entrenamiento.

### Modelo congelado

Entre las semillas de recompensa simple se eligió el mayor retorno en
validación, sujeto a que la magnitud del drawdown no superara 1,15 veces
la mediana del grupo.

| Semilla | Retorno de validación | Máximo drawdown |
|---|---:|---:|
| 42 | −69,46 % | −70,92 % |
| **123, seleccionada** | **−42,54 %** | **−48,69 %** |
| 2026 | −59,29 % | −61,55 % |

Checkpoint vigente: `modelo_congelado/modelo_final.pt`.

Configuración y selección:
`modelo_congelado/configuracion_final.json`.

SHA256:

```text
e9e7939454641152937860a55c86585b956b498a7bc3892f12aa47678fb3a525
```

El checkpoint de semilla 2026 del repositorio original de Actividad 4
pertenece a la versión anterior del experimento y no es el modelo final
de esta integración.

## 8. Actividad 5 — Evaluación y baselines

Se comparó DQN con comprar y mantener y una política aleatoria uniforme,
bajo las mismas condiciones de capital, costos y datos.

El held-out se evaluó en dos segmentos independientes:

| Segmento | Período UTC | Pasos |
|---|---|---:|
| 0 | 2022-06-09 00:00 a 2022-10-12 07:00 | 3.007 |
| 1 | 2022-10-13 12:00 a 2022-12-31 23:00 | 1.907 |

Cada segmento reinicia el capital en 100.000 y la posición en cero.
Sus retornos no se suman ni representan una inversión continua.

| Segmento | Estrategia | Retorno | Máximo drawdown | Costos |
|---|---|---:|---:|---:|
| 0 | DQN | −32,20 % | −34,37 % | 20.586,41 |
| 0 | Comprar y mantener | −36,84 % | −41,67 % | 99,90 |
| 0 | Aleatorio | −75,36 % | −75,53 % | 67.471,94 |
| 1 | DQN | −13,40 % | −23,18 % | 4.970,77 |
| 1 | Comprar y mantener | −10,09 % | −27,03 % | 99,90 |
| 1 | Aleatorio | −63,60 % | −64,85 % | 53.413,06 |

El DQN presentó menor máxima caída en ambos segmentos, pero no superó
consistentemente a comprar y mantener en retorno. Todas las estrategias
evaluadas perdieron capital.

La exposición real media del DQN fue 21,93 % y 8,13 %, respectivamente.
Su menor drawdown debe interpretarse junto con esa exposición reducida.

El baseline aleatorio utiliza una realización por segmento, con semillas
42 y 43. No se caracteriza toda su variabilidad.

### Archivos y análisis

- `src/baselines.py`: estrategias de referencia.
- `src/evaluate.py`: evaluación del modelo y baselines.
- `src/graficar_evaluacion.py`: generación de figuras.
- `modelo_congelado/`: checkpoint y configuración final.
- `evaluacion_validacion/`: comprobación del modelo en validación.
- `evaluacion_heldout/`: métricas, protocolo, trayectorias y seis figuras.
- [docs/resultados_actividad5.md](docs/resultados_actividad5.md):
  análisis completo, figuras, fallos identificados y limitaciones.

### Ejecución

Desde la raíz del proyecto:

```powershell
python -m src.evaluate --particion validacion
python -m src.evaluate --particion heldout
python -m src.graficar_evaluacion
```

Los comandos de evaluación requieren que sus carpetas de salida no
existan. Para reproducirlos, utilizar una copia limpia conservando
las evidencias originales.

El script de gráficos lee las trayectorias guardadas y no repite
el entrenamiento ni la evaluación.

## 9. Limitaciones y conclusión

El estudio utiliza un activo, un período histórico limitado y tres
semillas de entrenamiento. Las condiciones del mercado pueden cambiar.

Los huecos y el calentamiento reducen la cobertura temporal. Los
segmentos se evalúan con reinicios y sin liquidación final. La simulación
omite spread, deslizamiento e impacto de mercado.

Se identificaron pérdidas horarias importantes durante exposiciones
totales, pese a que el agente permaneció mayormente en efectivo.
La inactividad observada no demuestra por sí sola reward hacking.

No se demuestra superioridad consistente del DQN en rentabilidad.
El resultado se reporta sin reajustar el modelo con el held-out.

## Reproducción de datos y congelación del modelo

Los siguientes comandos reproducen dos procedimientos del proyecto sin
repetir el entrenamiento ni evaluar nuevamente el agente.

### Reconstrucción de datos corregidos

```powershell
python -m src.preparar_datos_corregidos
```

Lee los OHLCV de los CSV originales en `resultados/`, conserva los cortes
temporales originales y guarda en `resultados_corregidos_reproducidos/`.

Se verificó que los tres CSV reproducidos coinciden con los utilizados
por el modelo en fechas, columnas y valores, con tolerancias numéricas
`rtol=1e-10` y `atol=1e-12`.

### Congelación del modelo seleccionado

```powershell
python -m src.congelar_modelo
```

Aplica la regla documentada de selección de semilla para la recompensa
simple y guarda en `modelo_congelado_reproducido/`. La elección de la
recompensa se sustenta en la comparación A/B previamente documentada.

La reproducción seleccionó la semilla 123 y obtuvo el mismo SHA256
del checkpoint evaluado:

```text
e9e7939454641152937860a55c86585b956b498a7bc3892f12aa47678fb3a525
```

Ambos scripts requieren que sus carpetas de salida no existan.
Las carpetas oficiales del proyecto siguen siendo
`resultados_corregidos/` y `modelo_congelado/`.