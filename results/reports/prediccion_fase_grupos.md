# 🏆 Mundial 2026 — Predicción de la Fase de Grupos

**Método:** Regresión de Poisson (goles esperados λ por equipo) + corrección Dixon-Coles + Simulación de Monte Carlo: **10,000 iteraciones del torneo completo** — cada uno de los 72 partidos se simula 10,000 veces con perturbación estocástica de las condiciones del día (clima, estado físico, campo) e incentivos dinámicos en la jornada 3 (rotaciones de clasificados, empates que sirven a ambos, urgencias).

**Clasifican a 16avos:** los 2 primeros de cada grupo + los 8 mejores terceros de los 12 grupos.

El 🎯 Pronóstico es el resultado 1X2 más probable con su marcador más frecuente (útil para llenar la polla). Las probabilidades muestran cuán confiable es cada pronóstico.

## GRUPO A

### Partidos

| Jornada | Fecha | Partido | 🎯 Pronóstico | P(1) | P(X) | P(2) | Goles esperados |
|:-:|---|---|---|:-:|:-:|:-:|:-:|
| J1 | JUE 11/06 15:00 | **México** vs Sudáfrica | Gana México 2-0 | 76% | 18% | 6% | 2.2 - 0.5 |
| J1 | JUE 11/06 22:00 | **Corea del Sur** vs Rep. Checa | Gana Corea del Sur 1-0 | 44% | 29% | 27% | 1.4 - 1.1 |
| J2 | JUE 18/06 12:00 | **Rep. Checa** vs Sudáfrica | Gana Rep. Checa 1-0 | 52% | 26% | 22% | 1.7 - 1.0 |
| J2 | JUE 18/06 21:00 | **México** vs Corea del Sur | Gana México 1-0 | 59% | 25% | 16% | 1.7 - 0.8 |
| J3 | MIÉ 24/06 21:00 | Rep. Checa vs **México** | Gana México 0-1 | 12% | 23% | 65% | 0.7 - 1.9 |
| J3 | MIÉ 24/06 21:00 | Sudáfrica vs **Corea del Sur** | Gana Corea del Sur 0-2 | 17% | 24% | 59% | 0.9 - 1.9 |

P(1) = gana el primer equipo · P(X) = empate · P(2) = gana el segundo equipo.

### Tabla esperada del grupo

| Pos | Equipo | Pts esp. | GF-GC esp. | P(1°) | P(2°) | P(3° clasif.) | P(clasificar) |
|:-:|---|:-:|:-:|:-:|:-:|:-:|:-:|
| 1 | **México** | 6.6 | 5.9-2.0 | 68% | 22% | 6% | **97%** |
| 2 | **Corea del Sur** | 4.4 | 4.1-3.7 | 19% | 38% | 21% | **78%** |
| 3 | **Rep. Checa** | 3.5 | 3.4-4.3 | 10% | 27% | 26% | **63%** |
| 4 | Sudáfrica | 2.0 | 2.4-5.8 | 3% | 12% | 15% | **30%** |

---

## GRUPO B

### Partidos

| Jornada | Fecha | Partido | 🎯 Pronóstico | P(1) | P(X) | P(2) | Goles esperados |
|:-:|---|---|---|:-:|:-:|:-:|:-:|
| J1 | VIE 12/06 15:00 | **Canadá** vs Bosnia y Her. | Gana Canadá 2-0 | 70% | 21% | 9% | 2.0 - 0.6 |
| J1 | SÁB 13/06 15:00 | Catar vs **Suiza** | Gana Suiza 0-2 | 5% | 14% | 81% | 0.5 - 2.6 |
| J2 | JUE 18/06 15:00 | **Suiza** vs Bosnia y Her. | Gana Suiza 2-0 | 72% | 18% | 10% | 2.4 - 0.8 |
| J2 | JUE 18/06 18:00 | **Canadá** vs Catar | Gana Canadá 2-0 | 81% | 14% | 5% | 2.5 - 0.5 |
| J3 | MIÉ 24/06 15:00 | Suiza vs Canadá | Empate 1-1 | 33% | 31% | 36% | 1.1 - 1.1 |
| J3 | MIÉ 24/06 15:00 | **Bosnia y Her.** vs Catar | Gana Bosnia y Her. 1-0 | 53% | 26% | 21% | 1.8 - 1.1 |

P(1) = gana el primer equipo · P(X) = empate · P(2) = gana el segundo equipo.

### Tabla esperada del grupo

| Pos | Equipo | Pts esp. | GF-GC esp. | P(1°) | P(2°) | P(3° clasif.) | P(clasificar) |
|:-:|---|:-:|:-:|:-:|:-:|:-:|:-:|
| 1 | **Canadá** | 6.3 | 5.6-2.1 | 48% | 42% | 8% | **97%** |
| 2 | **Suiza** | 6.2 | 6.1-2.4 | 47% | 41% | 8% | **97%** |
| 3 | Bosnia y Her. | 2.8 | 3.1-5.4 | 4% | 13% | 33% | **49%** |
| 4 | Catar | 1.5 | 2.1-6.9 | 1% | 4% | 12% | **18%** |

---

## GRUPO C

### Partidos

| Jornada | Fecha | Partido | 🎯 Pronóstico | P(1) | P(X) | P(2) | Goles esperados |
|:-:|---|---|---|:-:|:-:|:-:|:-:|
| J1 | SÁB 13/06 18:00 | **Brasil** vs Marruecos | Gana Brasil 1-0 | 42% | 29% | 29% | 1.4 - 1.1 |
| J1 | SÁB 13/06 21:00 | Haití vs **Escocia** | Gana Escocia 0-1 | 21% | 27% | 52% | 0.9 - 1.6 |
| J2 | VIE 19/06 18:00 | Escocia vs **Marruecos** | Gana Marruecos 0-2 | 18% | 26% | 56% | 0.9 - 1.8 |
| J2 | VIE 19/06 20:30 | **Brasil** vs Haití | Gana Brasil 2-0 | 76% | 15% | 8% | 2.5 - 0.7 |
| J3 | MIÉ 24/06 18:00 | Escocia vs **Brasil** | Gana Brasil 0-2 | 15% | 22% | 63% | 0.9 - 2.1 |
| J3 | MIÉ 24/06 18:00 | **Marruecos** vs Haití | Gana Marruecos 2-0 | 69% | 21% | 11% | 2.2 - 0.7 |

P(1) = gana el primer equipo · P(X) = empate · P(2) = gana el segundo equipo.

### Tabla esperada del grupo

| Pos | Equipo | Pts esp. | GF-GC esp. | P(1°) | P(2°) | P(3° clasif.) | P(clasificar) |
|:-:|---|:-:|:-:|:-:|:-:|:-:|:-:|
| 1 | **Brasil** | 6.1 | 5.9-2.7 | 53% | 31% | 11% | **95%** |
| 2 | **Marruecos** | 5.4 | 5.1-3.0 | 36% | 39% | 15% | **90%** |
| 3 | **Escocia** | 3.3 | 3.4-4.8 | 9% | 21% | 29% | **59%** |
| 4 | Haití | 1.8 | 2.4-6.3 | 2% | 9% | 14% | **25%** |

---

## GRUPO D

### Partidos

| Jornada | Fecha | Partido | 🎯 Pronóstico | P(1) | P(X) | P(2) | Goles esperados |
|:-:|---|---|---|:-:|:-:|:-:|:-:|
| J1 | VIE 12/06 21:00 | EEUU vs Paraguay | Empate 1-1 | 37% | 30% | 34% | 1.2 - 1.1 |
| J1 | DOM 14/06 00:00 | Australia vs **Turquía** | Gana Turquía 0-1 | 32% | 29% | 38% | 1.2 - 1.3 |
| J2 | VIE 19/06 15:00 | EEUU vs Australia | Empate 1-1 | 35% | 28% | 37% | 1.3 - 1.3 |
| J2 | VIE 19/06 23:00 | **Turquía** vs Paraguay | Gana Turquía 1-0 | 41% | 28% | 31% | 1.4 - 1.2 |
| J3 | JUE 25/06 22:00 | **Turquía** vs EEUU | Gana Turquía 2-1 | 40% | 28% | 33% | 1.5 - 1.3 |
| J3 | JUE 25/06 22:00 | Paraguay vs **Australia** | Gana Australia 0-1 | 33% | 29% | 38% | 1.2 - 1.3 |

P(1) = gana el primer equipo · P(X) = empate · P(2) = gana el segundo equipo.

### Tabla esperada del grupo

| Pos | Equipo | Pts esp. | GF-GC esp. | P(1°) | P(2°) | P(3° clasif.) | P(clasificar) |
|:-:|---|:-:|:-:|:-:|:-:|:-:|:-:|
| 1 | **Turquía** | 4.4 | 4.3-3.7 | 31% | 26% | 17% | **75%** |
| 2 | **Australia** | 4.1 | 3.8-3.8 | 25% | 26% | 20% | **70%** |
| 3 | **EEUU** | 4.0 | 3.8-3.9 | 23% | 25% | 19% | **68%** |
| 4 | **Paraguay** | 3.8 | 3.5-3.9 | 21% | 23% | 20% | **64%** |

---

## GRUPO E

### Partidos

| Jornada | Fecha | Partido | 🎯 Pronóstico | P(1) | P(X) | P(2) | Goles esperados |
|:-:|---|---|---|:-:|:-:|:-:|:-:|
| J1 | DOM 14/06 13:00 | **Alemania** vs Curazao | Gana Alemania 2-0 | 76% | 16% | 7% | 2.5 - 0.7 |
| J1 | DOM 14/06 19:00 | Costa de Marfil vs **Ecuador** | Gana Ecuador 0-1 | 16% | 26% | 58% | 0.7 - 1.7 |
| J2 | SÁB 20/06 16:00 | **Alemania** vs Costa de Marfil | Gana Alemania 2-0 | 61% | 22% | 17% | 1.9 - 0.9 |
| J2 | SÁB 20/06 20:00 | **Ecuador** vs Curazao | Gana Ecuador 2-0 | 76% | 17% | 7% | 2.4 - 0.6 |
| J3 | JUE 25/06 16:00 | Curazao vs **Costa de Marfil** | Gana Costa de Marfil 0-2 | 18% | 22% | 59% | 1.0 - 2.0 |
| J3 | JUE 25/06 16:00 | Ecuador vs Alemania | Empate 1-1 | 37% | 29% | 34% | 1.2 - 1.2 |

P(1) = gana el primer equipo · P(X) = empate · P(2) = gana el segundo equipo.

### Tabla esperada del grupo

| Pos | Equipo | Pts esp. | GF-GC esp. | P(1°) | P(2°) | P(3° clasif.) | P(clasificar) |
|:-:|---|:-:|:-:|:-:|:-:|:-:|:-:|
| 1 | **Ecuador** | 5.9 | 5.4-2.6 | 44% | 37% | 13% | **94%** |
| 2 | **Alemania** | 5.8 | 5.7-2.8 | 45% | 36% | 13% | **94%** |
| 3 | **Costa de Marfil** | 3.5 | 3.6-4.6 | 10% | 21% | 34% | **64%** |
| 4 | Curazao | 1.5 | 2.3-7.0 | 2% | 6% | 12% | **19%** |

---

## GRUPO F

### Partidos

| Jornada | Fecha | Partido | 🎯 Pronóstico | P(1) | P(X) | P(2) | Goles esperados |
|:-:|---|---|---|:-:|:-:|:-:|:-:|
| J1 | DOM 14/06 16:00 | **Países Bajos** vs Japón | Gana Países Bajos 1-0 | 38% | 30% | 32% | 1.2 - 1.1 |
| J1 | DOM 14/06 22:00 | Suecia vs **Túnez** | Gana Túnez 0-1 | 32% | 29% | 39% | 1.2 - 1.4 |
| J2 | SÁB 20/06 13:00 | **Países Bajos** vs Suecia | Gana Países Bajos 2-0 | 74% | 17% | 9% | 2.5 - 0.8 |
| J2 | DOM 21/06 00:00 | Túnez vs **Japón** | Gana Japón 0-2 | 13% | 23% | 64% | 0.8 - 2.0 |
| J3 | JUE 25/06 19:00 | **Japón** vs Suecia | Gana Japón 2-0 | 69% | 19% | 12% | 2.3 - 0.8 |
| J3 | JUE 25/06 19:00 | Túnez vs **Países Bajos** | Gana Países Bajos 0-2 | 12% | 21% | 67% | 0.8 - 2.1 |

P(1) = gana el primer equipo · P(X) = empate · P(2) = gana el segundo equipo.

### Tabla esperada del grupo

| Pos | Equipo | Pts esp. | GF-GC esp. | P(1°) | P(2°) | P(3° clasif.) | P(clasificar) |
|:-:|---|:-:|:-:|:-:|:-:|:-:|:-:|
| 1 | **Países Bajos** | 6.0 | 5.9-2.7 | 51% | 34% | 9% | **95%** |
| 2 | **Japón** | 5.7 | 5.4-2.9 | 41% | 40% | 12% | **92%** |
| 3 | Túnez | 2.7 | 3.0-5.3 | 5% | 15% | 24% | **45%** |
| 4 | Suecia | 2.2 | 2.8-6.1 | 3% | 11% | 19% | **34%** |

---

## GRUPO G

### Partidos

| Jornada | Fecha | Partido | 🎯 Pronóstico | P(1) | P(X) | P(2) | Goles esperados |
|:-:|---|---|---|:-:|:-:|:-:|:-:|
| J1 | LUN 15/06 15:00 | **Bélgica** vs Egipto | Gana Bélgica 1-0 | 61% | 23% | 15% | 1.9 - 0.8 |
| J1 | LUN 15/06 21:00 | **Irán** vs Nueva Zelanda | Gana Irán 1-0 | 61% | 25% | 15% | 1.8 - 0.8 |
| J2 | DOM 21/06 15:00 | **Bélgica** vs Irán | Gana Bélgica 2-1 | 49% | 26% | 25% | 1.6 - 1.1 |
| J2 | DOM 21/06 21:00 | Nueva Zelanda vs **Egipto** | Gana Egipto 0-1 | 23% | 28% | 48% | 1.0 - 1.5 |
| J3 | VIE 26/06 23:00 | Egipto vs **Irán** | Gana Irán 0-1 | 24% | 29% | 47% | 1.0 - 1.5 |
| J3 | VIE 26/06 23:00 | Nueva Zelanda vs **Bélgica** | Gana Bélgica 0-2 | 10% | 17% | 73% | 0.8 - 2.5 |

P(1) = gana el primer equipo · P(X) = empate · P(2) = gana el segundo equipo.

### Tabla esperada del grupo

| Pos | Equipo | Pts esp. | GF-GC esp. | P(1°) | P(2°) | P(3° clasif.) | P(clasificar) |
|:-:|---|:-:|:-:|:-:|:-:|:-:|:-:|
| 1 | **Bélgica** | 6.2 | 6.0-2.8 | 59% | 26% | 10% | **95%** |
| 2 | **Irán** | 4.8 | 4.4-3.4 | 27% | 38% | 19% | **84%** |
| 3 | **Egipto** | 3.4 | 3.3-4.3 | 11% | 24% | 26% | **60%** |
| 4 | Nueva Zelanda | 2.1 | 2.6-5.8 | 4% | 12% | 16% | **31%** |

---

## GRUPO H

### Partidos

| Jornada | Fecha | Partido | 🎯 Pronóstico | P(1) | P(X) | P(2) | Goles esperados |
|:-:|---|---|---|:-:|:-:|:-:|:-:|
| J1 | LUN 15/06 12:00 | **España** vs Cabo Verde | Gana España 3-0 | 89% | 8% | 2% | 3.3 - 0.5 |
| J1 | LUN 15/06 18:00 | Arabia S. vs **Uruguay** | Gana Uruguay 0-1 | 13% | 23% | 65% | 0.7 - 1.9 |
| J2 | DOM 21/06 12:00 | **España** vs Arabia S. | Gana España 3-0 | 89% | 8% | 2% | 3.3 - 0.5 |
| J2 | DOM 21/06 18:00 | **Uruguay** vs Cabo Verde | Gana Uruguay 2-0 | 70% | 20% | 10% | 2.2 - 0.7 |
| J3 | VIE 26/06 20:00 | Cabo Verde vs **Arabia S.** | Gana Arabia S. 1-2 | 32% | 27% | 41% | 1.3 - 1.5 |
| J3 | VIE 26/06 20:00 | Uruguay vs **España** | Gana España 0-1 | 12% | 22% | 66% | 0.7 - 2.0 |

P(1) = gana el primer equipo · P(X) = empate · P(2) = gana el segundo equipo.

### Tabla esperada del grupo

| Pos | Equipo | Pts esp. | GF-GC esp. | P(1°) | P(2°) | P(3° clasif.) | P(clasificar) |
|:-:|---|:-:|:-:|:-:|:-:|:-:|:-:|
| 1 | **España** | 7.7 | 8.6-1.7 | 85% | 14% | 1% | **100%** |
| 2 | **Uruguay** | 5.0 | 4.7-3.4 | 14% | 65% | 11% | **90%** |
| 3 | Arabia S. | 2.3 | 2.7-6.5 | 1% | 12% | 22% | **35%** |
| 4 | Cabo Verde | 1.9 | 2.4-6.9 | 1% | 9% | 17% | **26%** |

---

## GRUPO I

### Partidos

| Jornada | Fecha | Partido | 🎯 Pronóstico | P(1) | P(X) | P(2) | Goles esperados |
|:-:|---|---|---|:-:|:-:|:-:|:-:|
| J1 | MAR 16/06 15:00 | **Francia** vs Senegal | Gana Francia 1-0 | 61% | 23% | 16% | 1.8 - 0.8 |
| J1 | MAR 16/06 18:00 | Irak vs **Noruega** | Gana Noruega 0-2 | 10% | 20% | 71% | 0.7 - 2.2 |
| J2 | LUN 22/06 17:00 | **Francia** vs Irak | Gana Francia 2-0 | 80% | 14% | 6% | 2.7 - 0.6 |
| J2 | LUN 22/06 20:00 | **Noruega** vs Senegal | Gana Noruega 1-0 | 50% | 26% | 24% | 1.7 - 1.1 |
| J3 | VIE 26/06 15:00 | Noruega vs **Francia** | Gana Francia 1-2 | 24% | 25% | 50% | 1.1 - 1.7 |
| J3 | VIE 26/06 15:00 | **Senegal** vs Irak | Gana Senegal 1-0 | 60% | 23% | 17% | 1.9 - 0.9 |

P(1) = gana el primer equipo · P(X) = empate · P(2) = gana el segundo equipo.

### Tabla esperada del grupo

| Pos | Equipo | Pts esp. | GF-GC esp. | P(1°) | P(2°) | P(3° clasif.) | P(clasificar) |
|:-:|---|:-:|:-:|:-:|:-:|:-:|:-:|
| 1 | **Francia** | 6.4 | 6.2-2.6 | 59% | 27% | 10% | **96%** |
| 2 | **Noruega** | 5.1 | 5.0-3.5 | 28% | 40% | 19% | **87%** |
| 3 | **Senegal** | 3.7 | 3.8-4.4 | 12% | 26% | 31% | **69%** |
| 4 | Irak | 1.5 | 2.2-6.7 | 2% | 6% | 11% | **19%** |

---

## GRUPO J

### Partidos

| Jornada | Fecha | Partido | 🎯 Pronóstico | P(1) | P(X) | P(2) | Goles esperados |
|:-:|---|---|---|:-:|:-:|:-:|:-:|
| J1 | MAR 16/06 21:00 | **Argentina** vs Argelia | Gana Argentina 2-0 | 65% | 23% | 12% | 1.9 - 0.7 |
| J1 | MIÉ 17/06 00:00 | **Austria** vs Jordania | Gana Austria 1-0 | 46% | 28% | 26% | 1.5 - 1.0 |
| J2 | LUN 22/06 13:00 | **Argentina** vs Austria | Gana Argentina 2-0 | 70% | 20% | 11% | 2.2 - 0.7 |
| J2 | LUN 22/06 23:00 | Jordania vs **Argelia** | Gana Argelia 0-1 | 25% | 27% | 48% | 1.1 - 1.6 |
| J3 | SÁB 27/06 22:00 | **Argelia** vs Austria | Gana Argelia 1-0 | 38% | 28% | 34% | 1.3 - 1.2 |
| J3 | SÁB 27/06 22:00 | Jordania vs **Argentina** | Gana Argentina 0-2 | 9% | 17% | 75% | 0.7 - 2.5 |

P(1) = gana el primer equipo · P(X) = empate · P(2) = gana el segundo equipo.

### Tabla esperada del grupo

| Pos | Equipo | Pts esp. | GF-GC esp. | P(1°) | P(2°) | P(3° clasif.) | P(clasificar) |
|:-:|---|:-:|:-:|:-:|:-:|:-:|:-:|
| 1 | **Argentina** | 6.9 | 6.6-2.2 | 75% | 18% | 5% | **98%** |
| 2 | **Argelia** | 3.7 | 3.6-4.2 | 12% | 34% | 21% | **67%** |
| 3 | **Austria** | 3.5 | 3.5-4.6 | 10% | 30% | 21% | **61%** |
| 4 | Jordania | 2.5 | 2.8-5.5 | 4% | 18% | 17% | **39%** |

---

## GRUPO K

### Partidos

| Jornada | Fecha | Partido | 🎯 Pronóstico | P(1) | P(X) | P(2) | Goles esperados |
|:-:|---|---|---|:-:|:-:|:-:|:-:|
| J1 | MIÉ 17/06 13:00 | **Portugal** vs RD Congo | Gana Portugal 2-0 | 66% | 22% | 12% | 1.9 - 0.7 |
| J1 | MIÉ 17/06 22:00 | Uzbekistán vs **Colombia** | Gana Colombia 0-2 | 13% | 23% | 63% | 0.8 - 1.9 |
| J2 | MAR 23/06 13:00 | **Portugal** vs Uzbekistán | Gana Portugal 2-0 | 61% | 24% | 16% | 1.9 - 0.9 |
| J2 | MAR 23/06 22:00 | **Colombia** vs RD Congo | Gana Colombia 2-0 | 71% | 19% | 10% | 2.2 - 0.7 |
| J3 | SÁB 27/06 19:30 | **Colombia** vs Portugal | Gana Colombia 2-1 | 41% | 27% | 32% | 1.5 - 1.3 |
| J3 | SÁB 27/06 19:30 | RD Congo vs **Uzbekistán** | Gana Uzbekistán 0-1 | 26% | 29% | 45% | 1.0 - 1.5 |

P(1) = gana el primer equipo · P(X) = empate · P(2) = gana el segundo equipo.

### Tabla esperada del grupo

| Pos | Equipo | Pts esp. | GF-GC esp. | P(1°) | P(2°) | P(3° clasif.) | P(clasificar) |
|:-:|---|:-:|:-:|:-:|:-:|:-:|:-:|
| 1 | **Colombia** | 5.9 | 5.6-2.8 | 50% | 33% | 10% | **93%** |
| 2 | **Portugal** | 5.5 | 5.1-3.0 | 39% | 39% | 13% | **91%** |
| 3 | **Uzbekistán** | 3.0 | 3.1-4.9 | 7% | 18% | 26% | **51%** |
| 4 | RD Congo | 2.1 | 2.5-5.6 | 4% | 10% | 18% | **32%** |

---

## GRUPO L

### Partidos

| Jornada | Fecha | Partido | 🎯 Pronóstico | P(1) | P(X) | P(2) | Goles esperados |
|:-:|---|---|---|:-:|:-:|:-:|:-:|
| J1 | MIÉ 17/06 16:00 | **Inglaterra** vs Croacia | Gana Inglaterra 1-0 | 54% | 27% | 19% | 1.6 - 0.9 |
| J1 | MIÉ 17/06 19:00 | Ghana vs **Panamá** | Gana Panamá 0-1 | 17% | 25% | 58% | 0.9 - 1.8 |
| J2 | MAR 23/06 16:00 | **Inglaterra** vs Ghana | Gana Inglaterra 2-0 | 84% | 12% | 5% | 2.9 - 0.5 |
| J2 | MAR 23/06 19:00 | Panamá vs **Croacia** | Gana Croacia 0-1 | 20% | 26% | 54% | 1.0 - 1.8 |
| J3 | SÁB 27/06 17:00 | Panamá vs **Inglaterra** | Gana Inglaterra 0-2 | 11% | 20% | 69% | 0.8 - 2.2 |
| J3 | SÁB 27/06 17:00 | **Croacia** vs Ghana | Gana Croacia 2-0 | 73% | 17% | 10% | 2.4 - 0.7 |

P(1) = gana el primer equipo · P(X) = empate · P(2) = gana el segundo equipo.

### Tabla esperada del grupo

| Pos | Equipo | Pts esp. | GF-GC esp. | P(1°) | P(2°) | P(3° clasif.) | P(clasificar) |
|:-:|---|:-:|:-:|:-:|:-:|:-:|:-:|
| 1 | **Inglaterra** | 6.8 | 6.7-2.2 | 66% | 25% | 7% | **98%** |
| 2 | **Croacia** | 5.1 | 5.1-3.4 | 25% | 46% | 18% | **89%** |
| 3 | **Panamá** | 3.4 | 3.6-4.9 | 7% | 23% | 32% | **62%** |
| 4 | Ghana | 1.5 | 2.2-7.1 | 1% | 6% | 10% | **18%** |
