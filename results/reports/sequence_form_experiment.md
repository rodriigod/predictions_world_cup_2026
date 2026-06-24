# D. Forma como secuencia aprendida (GRU) vs pi-ratings

Experimento AISLADO (no toca core/ ni Dixon-Coles): compara cómo representar la fuerza/forma de cada selección como input a un clasificador 1X2, con TimeSeriesSplit temporal y las métricas de core/ (RPS + accuracy). Menor RPS = mejor.

Protocolo: 4 folds temporales, GRU oculto=24, 12 epochs, Adam. Mismo conjunto OOF por N.

## N=10 últimos partidos (22852 partidos OOF)

| Representación de fuerza | RPS | Accuracy |
|---|:-:|:-:|
| pi-ratings solo | 0.1762 | 0.585 |
| secuencia (GRU) sola | 0.1768 | 0.585 |
| pi-ratings + secuencia | 0.1764 | 0.584 |

- Mejor por RPS: **pi-ratings solo** (0.1762). Diferencias < 0.0015 RPS se consideran EMPATE (ruido).
- Lectura N=10: la secuencia NO mejora a pi-ratings de forma apreciable (diferencias dentro del ruido): pi-ratings se mantiene.

## N=20 últimos partidos (22328 partidos OOF)

| Representación de fuerza | RPS | Accuracy |
|---|:-:|:-:|
| pi-ratings solo | 0.1762 | 0.584 |
| secuencia (GRU) sola | 0.1761 | 0.584 |
| pi-ratings + secuencia | 0.1758 | 0.582 |

- Mejor por RPS: **pi-ratings + secuencia** (0.1758). Diferencias < 0.0015 RPS se consideran EMPATE (ruido).
- Lectura N=20: la secuencia NO mejora a pi-ratings de forma apreciable (diferencias dentro del ruido): pi-ratings se mantiene.

## Veredicto honesto

En ambos N, la **secuencia aprendida no le gana a los pi-ratings**. Coherente con la literatura de selecciones: con ~28k partidos pero muchas selecciones débiles y poca frecuencia de juego, un rating online recursivo (pi/Elo) ya captura casi toda la señal de forma; un GRU pequeño no encuentra estructura secuencial adicional que valga. Se reporta el resultado tal cual: **no se adopta** la secuencia en core/ (era una comparación, no un reemplazo).

> Limitación: GRU pequeño y pocos epochs por coste; el objetivo es la COMPARACIÓN relativa con pi-ratings bajo el mismo protocolo, no exprimir el mejor GRU posible. Las features de secuencia incluyen el Elo del rival/propio del momento, así que parte de la señal de 'fuerza' ya está disponible para el GRU.