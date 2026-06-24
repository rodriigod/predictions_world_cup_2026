# C. Calibración de core (OOF, 384 partidos)

Reliability diagram: `calibration_core_oof.png`. Por clase, prob. predicha vs frecuencia real por bins. `gap = pred − obs`: **gap>0 = sobre-confianza** (predice más de lo que ocurre), gap<0 = sub-confianza.


## Clase 1 (gana local)

| p_pred | p_obs | gap (pred−obs) | n |
|:-:|:-:|:-:|:-:|
| 0.10 | 0.13 | -0.037 | 15 |
| 0.19 | 0.15 | +0.046 | 67 |
| 0.31 | 0.37 | -0.058 | 100 |
| 0.44 | 0.41 | +0.023 | 82 |
| 0.56 | 0.66 | -0.098 | 71 |
| 0.67 | 0.73 | -0.061 | 41 |
| 0.78 | 0.88 | -0.099 | 8 |

_Resumen 1 (gana local): sub-confianza en prob. altas._

## Clase X (empate)

| p_pred | p_obs | gap (pred−obs) | n |
|:-:|:-:|:-:|:-:|
| 0.12 | 0.00 | +0.124 | 1 |
| 0.21 | 0.13 | +0.083 | 106 |
| 0.29 | 0.27 | +0.022 | 277 |

_Resumen X (empate): razonablemente calibrada._

## Clase 2 (gana visita)

| p_pred | p_obs | gap (pred−obs) | n |
|:-:|:-:|:-:|:-:|
| 0.10 | 0.15 | -0.049 | 34 |
| 0.18 | 0.14 | +0.042 | 112 |
| 0.32 | 0.35 | -0.038 | 102 |
| 0.43 | 0.39 | +0.042 | 74 |
| 0.55 | 0.66 | -0.104 | 41 |
| 0.68 | 0.78 | -0.102 | 18 |
| 0.79 | 0.67 | +0.122 | 3 |

_Resumen 2 (gana visita): sub-confianza en prob. altas._

## Resumen global

- **ECE aproximado** (|gap| medio ponderado por bin/clase): **0.0490**.
- Diagnóstico, sin aplicar calibración todavía. Si la sobre-confianza en probabilidades altas es marcada, *temperature scaling* o *isotónica por clase* serían los candidatos naturales (el backtest de core ya reportó T≈1.0, así que se espera poca corrección).