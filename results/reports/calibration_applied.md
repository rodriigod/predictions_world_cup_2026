# Calibración aplicada — temperature scaling (384 partidos OOF)

T<1 = afilar (más confianza). Ajustado por NLL sobre OOF leak-free.

| Modelo | T* | RPS antes | RPS después | logloss antes | logloss después |
|---|:-:|:-:|:-:|:-:|:-:|
| solo core | 0.850 | 0.1994 | 0.1985 | 0.971 | 0.969 |
| stacking (OOF) | 1.184 | 0.2047 | 0.2050 | 0.998 | 0.995 |

## Chequeo held-out temporal (T ajustado en 60% antiguo, evaluado en 40% reciente)

| Modelo | T (train) | RPS sin calibrar | RPS calibrado |
|---|:-:|:-:|:-:|
| solo core | 0.779 | 0.2067 | 0.2071 |
| stacking (OOF) | 1.099 | 0.2112 | 0.2104 |

## Lectura

- core: T*=0.850 -> AFILAR (T<1): core está sub-confiado, calibrar ayuda.
- El T del ENSEMBLE (1.184) se guardó en `models/ensemble_temperature.json` y `predict_final` lo aplica a su salida (apply_calibration=True).
- La mejora de RPS suele ser pequeña (la calibración no cambia el argmax, solo la confianza); el valor está en log-loss/Brier y en que las probabilidades reflejen mejor la frecuencia real.