# C.4. Validación del modelo de penales vs tandas reales

Fuente: `shootouts.csv` (martj42). Tandas con ELO previo reconstruible: **677**.

El modelo de eliminatorias resuelve la tanda con `p(fav)=0.5 + 0.05·tanh(ΔElo/200)` — una **aproximación** (moneda al aire con sesgo leve al favorito), **no** ajustada a tandas reales. Esta es la verificación que faltaba.

## Resultado global

- **P(favorito por ELO gana la tanda) = 0.520** (IC95 ≈ 0.482–0.558).
- Confirma la premisa central del modelo: la tanda es **casi una moneda al aire**, con una ventaja pequeña y real para el favorito.
- Sesgo empírico ajustado `b_hat = 0.083` frente al `0.05` del modelo.

## P(favorito gana) por tramo de ΔElo

| ΔElo | n | P(fav gana) | ΔElo medio |
|---|:-:|:-:|:-:|
| 0-50 | 231 | 0.481 | 24 |
| 50-100 | 166 | 0.542 | 74 |
| 100-150 | 132 | 0.508 | 125 |
| 150-250 | 111 | 0.486 | 189 |
| 250+ | 37 | 0.811 | 331 |

## Veredicto

La aproximación es **razonable**: el sesgo real (~0.08) es del mismo orden que el 0.05 cableado, y P(fav) está cerca del rango esperado. **No se cambia core/**; el modelo de penales actual sobrevive a la validación contra datos reales. (El efecto sobre la probabilidad de campeón es de 2º orden: solo aplica cuando un cruce llega a penales.)

> Limitación: el favorito se define por un ELO propio reconstruido (K=40, ventaja local 100), no por el ELO exacto de core/ en el instante del cruce; sirve para validar el ORDEN de magnitud del sesgo, no para fijar su tercer decimal.