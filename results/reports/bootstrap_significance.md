# A. Significancia del backtest (bootstrap, N=2000)

Sobre **384 partidos OOF** del backtest de Mundiales. Remuestreo con reemplazo; menor RPS = mejor.

## RPS puntual por enfoque

| Enfoque | RPS |
|---|:-:|
| solo core | 0.1994 |
| promedio core+micro | 0.2005 |
| log-pooling | 0.2007 |
| core + Platt | 0.2024 |
| log-pooling extremizado | 0.2033 |
| core + isotonic | 0.2035 |
| stacking (OOF) | 0.2047 |
| solo microsim | 0.2054 |
| baseline FIFA ranking | 0.2204 |
| baseline uniforme | 0.2396 |

## Diferencia de RPS vs «solo core» (IC95%)

Positivo = peor que core. Si el IC incluye 0, no es concluyente.

| Enfoque | Δ media | IC95% | ¿incluye 0? |
|---|:-:|:-:|:-:|
| core + isotonic | +0.0041 | [-0.0012, +0.0093] | SÍ |
| core + Platt | +0.0029 | [+0.0004, +0.0054] | no |
| log-pooling | +0.0012 | [-0.0026, +0.0049] | SÍ |
| log-pooling extremizado | +0.0039 | [-0.0002, +0.0079] | SÍ |
| promedio core+micro | +0.0010 | [-0.0026, +0.0045] | SÍ |
| stacking (OOF) | +0.0052 | [-0.0005, +0.0109] | SÍ |
| solo microsim | +0.0059 | [-0.0012, +0.0130] | SÍ |
| baseline FIFA ranking | +0.0211 | [+0.0098, +0.0318] | no |
| baseline uniforme | +0.0403 | [+0.0264, +0.0535] | no |

## Veredicto

- El IC95% de (core + isotonic − core) **incluye 0** ([-0.0012, +0.0093]): la diferencia observada NO es estadísticamente concluyente — puede ser ruido de muestra.
- El IC95% de (core + Platt − core) **no incluye 0** ([+0.0004, +0.0054]): core + Platt es PEOR que core de forma estadísticamente significativa al 95%.
- El IC95% de (log-pooling − core) **incluye 0** ([-0.0026, +0.0049]): la diferencia observada NO es estadísticamente concluyente — puede ser ruido de muestra.
- El IC95% de (log-pooling extremizado − core) **incluye 0** ([-0.0002, +0.0079]): la diferencia observada NO es estadísticamente concluyente — puede ser ruido de muestra.
- El IC95% de (promedio core+micro − core) **incluye 0** ([-0.0026, +0.0045]): la diferencia observada NO es estadísticamente concluyente — puede ser ruido de muestra.
- El IC95% de (stacking (OOF) − core) **incluye 0** ([-0.0005, +0.0109]): la diferencia observada NO es estadísticamente concluyente — puede ser ruido de muestra.
- El IC95% de (solo microsim − core) **incluye 0** ([-0.0012, +0.0130]): la diferencia observada NO es estadísticamente concluyente — puede ser ruido de muestra.
- El IC95% de (baseline FIFA ranking − core) **no incluye 0** ([+0.0098, +0.0318]): baseline FIFA ranking es PEOR que core de forma estadísticamente significativa al 95%.
- El IC95% de (baseline uniforme − core) **no incluye 0** ([+0.0264, +0.0535]): baseline uniforme es PEOR que core de forma estadísticamente significativa al 95%.

> Lectura honesta: con solo unos cientos de partidos, las diferencias de RPS entre enfoques basados en fuerza de equipo suelen ser pequeñas frente al ruido de muestra. Un IC que incluye 0 significa que NO podemos afirmar con rigor que el stacking sea peor que core — solo que no lo mejora.