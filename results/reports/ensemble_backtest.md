# Ensemble — backtest comparativo (7 Mundiales 1998-2022)

OOF sobre **384 partidos**. Menor RPS/Brier/logloss = mejor. Brier = multiclase (Σ por clase, promedio por partido).

Calibración de core (isotónica/Platt) y pooling extremizado se miden OUT-OF-FOLD (el calibrador/`a` no ve la fila que evalúa).

| Enfoque | Accuracy | RPS | Brier | Log-loss |
|---|:-:|:-:|:-:|:-:|
| solo core | 0.547 | 0.1994 | 0.5737 | 0.971 |
| solo microsim | 0.536 | 0.2054 | 0.5859 | 0.994 |
| promedio core+micro | 0.536 | 0.2005 | 0.5751 | 0.974 |
| stacking (OOF) | 0.521 | 0.2047 | 0.5860 | 0.998 |
| core + isotonic | 0.544 | 0.2035 | 0.5841 | 1.246 |
| core + Platt | 0.552 | 0.2024 | 0.5799 | 0.986 |
| log-pooling | 0.536 | 0.2007 | 0.5756 | 0.975 |
| log-pooling extremizado | 0.536 | 0.2033 | 0.5810 | 0.986 |
| baseline FIFA ranking | 0.542 | 0.2204 | 0.6182 | 1.030 |
| baseline uniforme | 0.435 | 0.2396 | 0.6667 | 1.099 |

_Log-pooling extremizado: a global = 1.00 (neutro, elegido por CV minimizando RPS)._

Meta-modelo: logística multinomial L2, **C=0.1** (CV temporal).

## Coeficientes (escala estandarizada)

| Feature | Grupo | coef_1 | coef_X | coef_2 | \|peso\| |
|---|:-:|:-:|:-:|:-:|:-:|
| p_away_core | core | -0.226 | -0.048 | 0.274 | 0.183 |
| p_home_core | core | 0.228 | 0.016 | -0.244 | 0.162 |
| p_draw_core | core | -0.075 | 0.124 | -0.048 | 0.082 |
| p_home_micro | microsim | 0.050 | -0.045 | -0.005 | 0.033 |
| p_away_micro | microsim | -0.045 | 0.041 | 0.004 | 0.030 |
| p_draw_micro | microsim | -0.029 | 0.025 | 0.004 | 0.019 |
| consenso_home | llm | 0.000 | 0.000 | 0.000 | 0.000 |
| fatiga_home | llm | 0.000 | 0.000 | 0.000 | 0.000 |
| consenso_away_missing | llm | 0.000 | 0.000 | 0.000 | 0.000 |
| consenso_home_missing | llm | 0.000 | 0.000 | 0.000 | 0.000 |
| consenso_away | llm | 0.000 | 0.000 | 0.000 | 0.000 |
| cambio_dt_home | llm | 0.000 | 0.000 | 0.000 | 0.000 |
| dead_rubber | llm | 0.000 | 0.000 | 0.000 | 0.000 |
| cambio_dt_away | llm | 0.000 | 0.000 | 0.000 | 0.000 |
| lesionados_away | llm | 0.000 | 0.000 | 0.000 | 0.000 |
| lesionados_home | llm | 0.000 | 0.000 | 0.000 | 0.000 |
| fatiga_away | llm | 0.000 | 0.000 | 0.000 | 0.000 |

## Importancia por grupo (suma \|peso\|)

- **core**: 0.428
- **microsim**: 0.083
- **llm**: 0.000

> Nota honesta: las señales del LLM entran NEUTRAS en el backtest histórico (no hay forma leak-free de reconstruir lesiones/consenso de 1998-2022), así que sus coeficientes salen ~0 por FALTA DE SEÑAL EN ENTRENAMIENTO, no por medirse inútiles. Quedan cableadas para la inferencia 2026 (datos reales).