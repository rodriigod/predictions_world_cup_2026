# A. Calibración de core — isotónica vs Platt (384 partidos OOF)

Evaluación OUT-OF-FOLD (StratifiedKFold): el calibrador no ve la fila que mide, así no se sobre-ajusta la calibración. Menor RPS/Brier/logloss = mejor.

| Método | RPS | Brier | Log-loss | Accuracy |
|---|:-:|:-:|:-:|:-:|
| core sin calibrar | 0.1994 | 0.5737 | 0.971 | 0.547 |
| core + isotónica | 0.2035 | 0.5841 | 1.246 | 0.544 |
| core + Platt | 0.2024 | 0.5799 | 0.986 | 0.552 |

## Decisión

**No gana ninguno**: ni isotónica ni Platt mejoran RPS y Brier a la vez frente a core sin calibrar sobre held-out OOF. No se guarda calibrador; predict_final deja core SIN calibrar (`calibrate_core=False`, por defecto). Coherente con que el temperature scaling ya había dado T≈1 y mejoras marginales: calibrar core no aporta señal real en este conjunto.

> Nota metodológica: Platt = logística multinomial sobre el logit de las probs (generaliza temperature scaling, que es el caso de 1 parámetro). Isotónica = monótona no paramétrica por clase + renormalización; con ~344 partidos es la más expuesta a overfit, por eso la evaluación CV-OOF es imprescindible.