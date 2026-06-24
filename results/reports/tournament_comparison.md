# Simulación del torneo — 6 modelos (campeón %)

Monte Carlo, 1500 torneos por modelo. Cuadro oficial 2026 (12 grupos, avanzan 1º/2º + 8 mejores 3º). `sin` = ratings pre-torneo; `con` = ratings online-updated con la 1ª ronda real.

## Probabilidad de ser CAMPEÓN (top 16)

| Equipo | core·sin | core·con | micr·sin | micr·con | ense·sin | ense·con |
|---|:-:|:-:|:-:|:-:|:-:|:-:|
| España | 20.5 | 15.0 | 33.6 | 29.2 | 17.1 | 14.3 |
| Argentina | 16.0 | 20.2 | 26.9 | 29.1 | 17.2 | 18.7 |
| Francia | 8.7 | 11.3 | 11.5 | 14.7 | 9.9 | 12.4 |
| Inglaterra | 6.9 | 9.5 | 6.5 | 8.1 | 7.5 | 8.5 |
| Colombia | 6.3 | 7.7 | 3.5 | 3.9 | 6.7 | 6.9 |
| Brasil | 7.1 | 7.0 | 3.4 | 3.4 | 6.7 | 5.8 |
| México | 4.5 | 3.6 | 0.6 | 0.6 | 3.8 | 4.9 |
| Portugal | 4.5 | 3.7 | 3.4 | 1.7 | 3.8 | 3.8 |
| Noruega | 1.8 | 3.1 | 0.1 | 0.7 | 2.2 | 2.3 |
| Países Bajos | 1.8 | 1.9 | 0.9 | 1.1 | 3.0 | 2.1 |
| Alemania | 2.1 | 3.0 | 1.3 | 2.0 | 2.7 | 2.6 |
| Marruecos | 2.7 | 1.9 | 1.1 | 0.9 | 2.4 | 2.6 |
| Ecuador | 2.3 | 1.4 | 2.1 | 0.8 | 2.6 | 2.3 |
| Bélgica | 2.3 | 1.9 | 0.7 | 0.5 | 2.3 | 2.1 |
| Suiza | 2.1 | 1.6 | 0.2 | 0.1 | 2.2 | 1.5 |
| Canadá | 1.9 | 1.0 | 0.1 | 0.0 | 0.9 | 1.0 |

## Cómo leerlo

- Cada columna es un modelo. `core·sin` vs `core·con` = efecto de incorporar la 1ª ronda real en core; igual para microsim y ensemble.
- microsim suele ser más extremo (favorece más a los fuertes); ensemble modera; core está en medio.
- Las tablas por ronda (16avos→final) están en `results/reports/tournament_sim/<modelo>_<regimen>.csv`.

> 1500 simulaciones: hay ruido de muestreo de ~±1pp en las probabilidades de campeón. Para diferencias finas, subir N_SIMS.