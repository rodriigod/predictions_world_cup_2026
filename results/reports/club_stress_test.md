# D. Stress-test de stacking en liga de clubes (alto volumen)

Diagnóstico de metodología (NO producción del Mundial). Pipeline paralelo en `experiments/club_stress_test/`.

Datos: **36,708 partidos** OOF de top-5 ligas europeas (2000-2025). Dos modelos base de fuerza (A=ELO, B=goles+forma) + meta-logística.

| Enfoque | Accuracy | RPS | Log-loss |
|---|:-:|:-:|:-:|
| solo A (ELO) | 0.521 | 0.2005 | 0.989 |
| solo B (goles+forma) | 0.499 | 0.2088 | 1.015 |
| promedio A+B | 0.517 | 0.2025 | 0.995 |
| stacking (OOF) | 0.522 | 0.2006 | 0.989 |

Mejor base single: **solo A (ELO)**. stacking − solo A (ELO): ΔRPS = **+0.0000**, IC95% = [-0.0000, +0.0001] (incluye 0: SÍ).

## Conclusión

**El stacking NO le gana al mejor modelo base ni con decenas de miles de partidos.** Refuerza que el problema es CONCEPTUAL: cuando los modelos base comparten la señal dominante (fuerza de equipo), apilarlos agrega varianza, no skill — el tamaño de muestra del Mundial no es la causa principal.

> Nota: el meta-modelo se entrena con las predicciones base in-sample del tramo de train (stacking 'con refit'), un sesgo que solo FAVORECE al stacking; aun así el resultado de arriba manda.