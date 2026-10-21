# Actividad 5 — Evaluación y baselines

**Materia:** Simulación y Aprendizaje por Refuerzo  
**Proyecto:** Agente inversor con DQN  
**Responsable:** Mauricio Plascencia Islas

## 1. Objetivo

Evaluar un agente DQN congelado sobre datos no utilizados para su entrenamiento ni selección, comparándolo con comprar y mantener y una política aleatoria.

La comparación considera retorno acumulado, máximo drawdown, costos de transacción, operaciones y comportamiento de las posiciones.

## 2. Modelo evaluado

El agente utiliza cuatro entradas: retorno horario, distancia a la media móvil de 24 horas, volatilidad de 24 retornos horarios y posición real actual.

Sus acciones representan objetivos de inversión de 0 %, 50 % o 100 % del capital después de costos.

- Arquitectura: red neuronal 64–64, activaciones ReLU y tres salidas Q.
- Recompensa seleccionada: retorno simple neto.
- Semilla del modelo congelado: 123.
- Entrenamiento: 30.000 pasos.
- Checkpoint: `modelo_congelado/modelo_final.pt`.
- Configuración: `modelo_congelado/configuracion_final.json`.

Huella SHA256 del checkpoint:

```text
e9e7939454641152937860a55c86585b956b498a7bc3892f12aa47678fb3a525
```

La recompensa se eligió mediante comparación en validación de las variantes simple y logarítmica, con semillas 42, 123 y 2026. Ambas obtuvieron retornos medios prácticamente iguales; se conservó la simple por su menor dispersión, menor máxima caída promedio y menores costos promedio. Con tres semillas no se afirma superioridad estadística.

Entre los modelos de recompensa simple se seleccionó el de mayor retorno de validación, sujeto a que la magnitud de su máximo drawdown no excediera 1,15 veces la mediana del grupo.

El modelo seleccionado obtuvo en validación un retorno de −42,54 % y un máximo drawdown de −48,69 %. Sus pesos y configuración se mantuvieron fijos durante la evaluación held-out.

## 3. Datos y protocolo de evaluación

Se utilizaron precios horarios de BTC/USDT provenientes de Bitfinex2 y los datos de `resultados_corregidos/heldout.csv`.

Se conservaron las fronteras temporales originales de entrenamiento, validación y held-out. El escalado se ajustó exclusivamente con entrenamiento.

Los huecos temporales se trataron sin imputar precios: los indicadores se recalcularon por tramos horarios continuos y se descartaron 24 observaciones de calentamiento al inicio de cada tramo.

El held-out quedó dividido en dos segmentos:

| Segmento | Inicio UTC | Fin UTC | Transiciones horarias |
|---|---|---|---:|
| 0 | 2022-06-09 00:00 | 2022-10-12 07:00 | 3.007 |
| 1 | 2022-10-13 12:00 | 2022-12-31 23:00 | 1.907 |

Cada estrategia comenzó cada segmento con:

- Capital de 100.000 unidades monetarias.
- Posición inicial de 0 %.
- Comisión del 0,1 % sobre el importe negociado.
- Las mismas observaciones y oportunidades de decisión.
- Sin liquidación forzada al finalizar.

Los segmentos son evaluaciones independientes, con reinicio de capital y posición. Sus retornos no se suman ni se presentan como una inversión continua durante todo el held-out.

La decisión se toma con información hasta el cierre de t. El entorno aplica la operación y su costo antes del movimiento de precio de t a t+1.

Se supone ejecución al cierre observado, sin deslizamiento, diferencial de compra y venta, apalancamiento ni ventas en corto. El efectivo no genera intereses.

## 4. Estrategias comparadas

### 4.1. DQN

Utiliza el checkpoint congelado y selecciona la acción con mayor valor Q, sin exploración ni actualización de pesos.

### 4.2. Comprar y mantener

Selecciona el objetivo 100 % durante todo el segmento. Compra al inicio y mantiene las unidades del activo; solo paga la comisión de entrada.

### 4.3. Agente aleatorio

Selecciona uniformemente entre las acciones 0, 1 y 2.

Se utilizaron semillas predeterminadas:

- Segmento 0: 42.
- Segmento 1: 43.

Se evaluó una realización aleatoria por segmento. Por ello, los resultados no caracterizan toda la variabilidad posible de este baseline.

## 5. Métricas

El retorno acumulado se calculó como:

`capital_final / capital_inicial − 1`

El drawdown en cada instante se calculó como:

`capital_actual / máximo_previo_del_capital − 1`

El máximo drawdown corresponde al mínimo de esa serie y se reporta con signo negativo.

Los costos son la suma de las comisiones efectivamente pagadas. Una operación se cuenta cuando existe un importe negociado distinto de cero, según la tolerancia numérica del entorno.

Las proporciones de acciones describen las decisiones objetivo. La exposición real media corresponde al promedio de la posición al final de cada paso.

## 6. Resultados held-out

| Segmento | Estrategia | Capital final | Retorno | Máximo drawdown | Costos | Operaciones |
|---|---|---:|---:|---:|---:|---:|
| 0 | DQN | 67.802,35 | −32,20 % | −34,37 % | 20.586,41 | 302 |
| 0 | Comprar y mantener | 63.162,98 | −36,84 % | −41,67 % | 99,90 | 1 |
| 0 | Aleatorio | 24.638,53 | −75,36 % | −75,53 % | 67.471,94 | 2.324 |
| 1 | DQN | 86.597,68 | −13,40 % | −23,18 % | 4.970,77 | 72 |
| 1 | Comprar y mantener | 89.906,83 | −10,09 % | −27,03 % | 99,90 | 1 |
| 1 | Aleatorio | 36.401,48 | −63,60 % | −64,85 % | 53.413,06 | 1.443 |

### 6.1. Segmento 0

El DQN perdió menos capital que comprar y mantener: la diferencia de retorno fue de aproximadamente 4,64 puntos porcentuales.

Su máxima caída fue menor en aproximadamente 7,31 puntos porcentuales. Sin embargo, sufrió una pérdida importante al inicio y terminó por debajo del capital inicial.

![Capital y drawdown del segmento 0](../evaluacion_heldout/figuras/capital_drawdown_segmento_0.png)

### 6.2. Segmento 1

El DQN obtuvo un retorno inferior al de comprar y mantener en aproximadamente 3,31 puntos porcentuales, aunque su máxima caída fue menor en aproximadamente 3,85 puntos porcentuales.

Esto muestra que una menor caída desde máximos anteriores no garantiza un mayor capital final. Cada estrategia calcula su drawdown respecto a su propia trayectoria de máximos.

![Capital y drawdown del segmento 1](../evaluacion_heldout/figuras/capital_drawdown_segmento_1.png)

## 7. Decisiones y exposición del DQN

| Indicador | Segmento 0 | Segmento 1 |
|---|---:|---:|
| Objetivo 0 % | 77,39 % | 91,40 % |
| Objetivo 50 % | 1,36 % | 0,94 % |
| Objetivo 100 % | 21,25 % | 7,66 % |
| Exposición real media al final del paso | 21,93 % | 8,13 % |

El agente permaneció mayormente en efectivo y utilizó poco la acción intermedia. Los períodos sin exposición explican los tramos horizontales de la curva de capital.

La exposición reducida debe considerarse al interpretar su menor drawdown. Los resultados no demuestran por sí solos capacidad de anticipar movimientos del mercado ni permiten atribuir la mejora a una predicción acertada.

Las acciones se muestran asociadas a la fecha de su resultado en t+1.

![Acciones y posiciones del segmento 0](../evaluacion_heldout/figuras/acciones_posiciones_DQN_segmento_0.png)

![Acciones y posiciones del segmento 1](../evaluacion_heldout/figuras/acciones_posiciones_DQN_segmento_1.png)

## 8. Costos de transacción

Comprar y mantener pagó únicamente la entrada: 99,90 unidades monetarias por segmento.

El DQN pagó menos costos que la realización aleatoria evaluada, pero considerablemente más que comprar y mantener. En el segmento 1, parte del aumento de comisiones coincide visualmente con la concentración de operaciones de noviembre.

El número de operaciones no determina por sí solo los costos: también importan el importe negociado y el capital disponible en cada momento.

Los costos ya están descontados de las curvas de capital. No deben descontarse nuevamente ni sumarse al capital final para inferir una trayectoria sin comisiones.

![Costos del segmento 0](../evaluacion_heldout/figuras/costos_segmento_0.png)

![Costos del segmento 1](../evaluacion_heldout/figuras/costos_segmento_1.png)

## 9. Ejemplos de comportamiento problemático

Se identificaron los siguientes peores pasos horarios del DQN:

| Segmento | Fecha del resultado UTC | Retorno neto horario | Objetivo elegido | Comisión del paso |
|---|---|---:|---:|---:|
| 0 | 2022-06-14 01:00 | −4,20 % | 100 % | 41,59 |
| 1 | 2022-11-08 17:00 | −5,07 % | 100 % | 0,00 |

En el segundo caso, el agente mantuvo exposición total durante una caída importante y no pagó comisión en ese paso. La pérdida corresponde al movimiento del activo mientras estaba invertido; no implica que hubiera abierto la posición justo antes.

Este episodio evidencia que permanecer mayormente en efectivo no garantiza evitar pérdidas en las ocasiones en que el agente entra o permanece en el mercado.

No se ha demostrado que la inactividad observada constituya reward hacking. Su interpretación requiere distinguir entre una respuesta a los costos, las condiciones de mercado y las limitaciones del aprendizaje.

## 10. Limitaciones

- Se estudió un solo activo y un período histórico limitado; los mercados son no estacionarios.
- Solo se utilizaron tres semillas para comparar recompensas y seleccionar el modelo.
- El baseline aleatorio tiene una realización por segmento.
- La exclusión de huecos y períodos de calentamiento limita la cobertura temporal.
- Los reinicios impiden interpretar los segmentos como una única inversión continua.
- La ejecución al cierre observado es una simplificación; no se modelaron deslizamiento, spread, impacto de mercado ni restricciones de liquidez.
- No se realizó liquidación final, por lo que el capital final representa la valoración del portafolio y no necesariamente efectivo después de vender.
- La menor exposición del DQN dificulta atribuir su menor drawdown a una mejor selección de momentos de inversión.
- Las observaciones del held-out se utilizaron para análisis final, sin reajustar el modelo.

## 11. Reproducibilidad y archivos

Código principal:

- `src/baselines.py`
- `src/evaluate.py`
- `src/graficar_evaluacion.py`
- `src/agent.py`
- `src/environment.py`
- `src/features.py`
- `src/rewards.py`

Modelo y datos:

- `modelo_congelado/modelo_final.pt`
- `modelo_congelado/configuracion_final.json`
- `modelo_congelado/resumen_datos.json`
- `resultados_corregidos/validacion.csv`
- `resultados_corregidos/heldout.csv`

Evidencias:

- `evaluacion_validacion/metricas.csv`
- `evaluacion_heldout/metricas.csv`
- `evaluacion_heldout/protocolo.json`
- Seis CSV de trayectorias held-out.
- Seis figuras en `evaluacion_heldout/figuras/`.

Comandos desde la raíz del proyecto:

```powershell
python -m pytest .\tests -q
python -m src.evaluate --particion validacion
python -m src.evaluate --particion heldout
python -m src.graficar_evaluacion
```

Los comandos de evaluación requieren que sus carpetas de salida no existan, como protección frente a sobrescrituras. Para reproducirlos, utilizar una copia limpia del proyecto sin esas carpetas generadas, conservando las evidencias originales.

La última ejecución de la suite obtuvo 42 pruebas aprobadas y dos
advertencias sobre límites infinitos del espacio de observación.
Estas pruebas no incluyen comprobaciones automatizadas específicas
de los módulos de evaluación. Estos módulos se verificaron mediante
su ejecución y la reproducción del retorno y drawdown de validación
del checkpoint seleccionado.

La evaluación en validación reprodujo el retorno y drawdown registrados para el checkpoint seleccionado.

## 12. Conclusión

El DQN superó a comprar y mantener en retorno en el segmento 0, pero obtuvo un retorno inferior en el segmento 1. Presentó menor máximo drawdown que esa estrategia en ambos segmentos y superó a la realización aleatoria evaluada.

Todas las estrategias comparadas perdieron capital en ambos segmentos. Bajo el supuesto de efectivo sin intereses ni costos de tenencia, permanecer siempre en efectivo habría conservado el capital inicial.

Por tanto, no se demuestra una superioridad consistente del DQN en rentabilidad. El aporte del experimento es una evaluación temporal explícita, con costos, baselines, modelo congelado y análisis de sus fallos.

**Idea para recordar:** evaluar un agente implica comparar rentabilidad, riesgo, exposición y costos; una menor caída o superar a un agente aleatorio no bastan para demostrar una estrategia de inversión superior.    