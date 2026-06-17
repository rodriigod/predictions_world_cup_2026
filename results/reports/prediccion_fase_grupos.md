# 🏆 Mundial 2026 — Predicción de la Fase de Grupos

**Método:** Regresión de Poisson (goles esperados λ por equipo) + corrección Dixon-Coles + Simulación de Monte Carlo: **50,000 iteraciones del torneo completo** — cada uno de los 72 partidos se simula 50,000 veces con perturbación estocástica de las condiciones del día (clima, estado físico, campo) e incentivos dinámicos en la jornada 3 (rotaciones de clasificados, empates que sirven a ambos, urgencias).

**Clasifican a 16avos:** los 2 primeros de cada grupo + los 8 mejores terceros de los 12 grupos.

El 🎯 Pronóstico es el resultado 1X2 más probable con su marcador más frecuente (útil para llenar la polla). Las probabilidades muestran cuán confiable es cada pronóstico.

## GRUPO A

### Partidos

| Jornada | Fecha | Partido | 🎯 Pronóstico | P(1) | P(X) | P(2) | Goles esperados |
|:-:|---|---|---|:-:|:-:|:-:|:-:|
| J1 | JUE 11/06 15:00 | **México** vs Sudáfrica | Gana México 2-0 | 78% | 16% | 6% | 2.3 - 0.5 |
| J1 | JUE 11/06 22:00 | **Corea del Sur** vs Rep. Checa | Gana Corea del Sur 1-0 | 44% | 27% | 28% | 1.5 - 1.2 |
| J2 | JUE 18/06 12:00 | **Rep. Checa** vs Sudáfrica | Gana Rep. Checa 2-1 | 56% | 24% | 20% | 1.9 - 1.0 |
| J2 | JUE 18/06 21:00 | **México** vs Corea del Sur | Gana México 1-0 | 59% | 25% | 17% | 1.8 - 0.8 |
| J3 | MIÉ 24/06 21:00 | Rep. Checa vs **México** | Gana México 0-2 | 13% | 21% | 66% | 0.8 - 2.1 |
| J3 | MIÉ 24/06 21:00 | Sudáfrica vs **Corea del Sur** | Gana Corea del Sur 0-2 | 16% | 21% | 63% | 0.9 - 2.1 |

P(1) = gana el primer equipo · P(X) = empate · P(2) = gana el segundo equipo.

### Tabla esperada del grupo

| Pos | Equipo | Pts esp. | GF-GC esp. | P(1°) | P(2°) | P(3° clasif.) | P(clasificar) |
|:-:|---|:-:|:-:|:-:|:-:|:-:|:-:|
| 1 | **México** | 6.7 | 6.2-2.2 | 68% | 22% | 7% | **97%** |
| 2 | **Corea del Sur** | 4.5 | 4.4-3.9 | 19% | 39% | 22% | **79%** |
| 3 | **Rep. Checa** | 3.6 | 3.8-4.7 | 11% | 29% | 26% | **65%** |
| 4 | Sudáfrica | 1.9 | 2.5-6.3 | 2% | 10% | 13% | **26%** |

---

## GRUPO B

### Partidos

| Jornada | Fecha | Partido | 🎯 Pronóstico | P(1) | P(X) | P(2) | Goles esperados |
|:-:|---|---|---|:-:|:-:|:-:|:-:|
| J1 | VIE 12/06 15:00 | **Canadá** vs Bosnia y Her. | Gana Canadá 2-0 | 70% | 21% | 8% | 2.0 - 0.5 |
| J1 | SÁB 13/06 15:00 | Catar vs **Suiza** | Gana Suiza 0-2 | 4% | 11% | 85% | 0.6 - 3.0 |
| J2 | JUE 18/06 15:00 | **Suiza** vs Bosnia y Her. | Gana Suiza 2-0 | 75% | 17% | 8% | 2.5 - 0.7 |
| J2 | JUE 18/06 18:00 | **Canadá** vs Catar | Gana Canadá 2-0 | 84% | 12% | 4% | 2.8 - 0.5 |
| J3 | MIÉ 24/06 15:00 | Suiza vs **Canadá** | Gana Canadá 0-1 | 33% | 30% | 37% | 1.1 - 1.2 |
| J3 | MIÉ 24/06 15:00 | **Bosnia y Her.** vs Catar | Gana Bosnia y Her. 2-1 | 54% | 24% | 21% | 1.9 - 1.1 |

P(1) = gana el primer equipo · P(X) = empate · P(2) = gana el segundo equipo.

### Tabla esperada del grupo

| Pos | Equipo | Pts esp. | GF-GC esp. | P(1°) | P(2°) | P(3° clasif.) | P(clasificar) |
|:-:|---|:-:|:-:|:-:|:-:|:-:|:-:|
| 1 | **Suiza** | 6.4 | 6.6-2.5 | 48% | 42% | 7% | **97%** |
| 2 | **Canadá** | 6.4 | 6.0-2.2 | 48% | 43% | 7% | **97%** |
| 3 | Bosnia y Her. | 2.8 | 3.2-5.6 | 3% | 12% | 33% | **48%** |
| 4 | Catar | 1.4 | 2.2-7.7 | 1% | 4% | 11% | **15%** |

---

## GRUPO C

### Partidos

| Jornada | Fecha | Partido | 🎯 Pronóstico | P(1) | P(X) | P(2) | Goles esperados |
|:-:|---|---|---|:-:|:-:|:-:|:-:|
| J1 | SÁB 13/06 18:00 | **Brasil** vs Marruecos | Gana Brasil 1-0 | 44% | 28% | 27% | 1.4 - 1.1 |
| J1 | SÁB 13/06 21:00 | Haití vs **Escocia** | Gana Escocia 0-1 | 22% | 26% | 53% | 1.0 - 1.7 |
| J2 | VIE 19/06 18:00 | Escocia vs **Marruecos** | Gana Marruecos 0-1 | 20% | 25% | 54% | 1.0 - 1.7 |
| J2 | VIE 19/06 20:30 | **Brasil** vs Haití | Gana Brasil 2-0 | 79% | 14% | 7% | 2.8 - 0.8 |
| J3 | MIÉ 24/06 18:00 | Escocia vs **Brasil** | Gana Brasil 1-2 | 15% | 20% | 64% | 1.0 - 2.3 |
| J3 | MIÉ 24/06 18:00 | **Marruecos** vs Haití | Gana Marruecos 2-0 | 69% | 19% | 12% | 2.3 - 0.8 |

P(1) = gana el primer equipo · P(X) = empate · P(2) = gana el segundo equipo.

### Tabla esperada del grupo

| Pos | Equipo | Pts esp. | GF-GC esp. | P(1°) | P(2°) | P(3° clasif.) | P(clasificar) |
|:-:|---|:-:|:-:|:-:|:-:|:-:|:-:|
| 1 | **Brasil** | 6.3 | 6.6-2.9 | 57% | 29% | 10% | **96%** |
| 2 | **Marruecos** | 5.3 | 5.1-3.2 | 32% | 40% | 16% | **89%** |
| 3 | **Escocia** | 3.4 | 3.7-5.0 | 9% | 22% | 28% | **59%** |
| 4 | Haití | 1.8 | 2.7-6.9 | 2% | 8% | 13% | **24%** |

---

## GRUPO D

### Partidos

| Jornada | Fecha | Partido | 🎯 Pronóstico | P(1) | P(X) | P(2) | Goles esperados |
|:-:|---|---|---|:-:|:-:|:-:|:-:|
| J1 | VIE 12/06 21:00 | EEUU vs Paraguay | Empate 1-1 | 37% | 29% | 34% | 1.3 - 1.2 |
| J1 | DOM 14/06 00:00 | Australia vs Turquía | Empate 1-1 | 35% | 29% | 37% | 1.2 - 1.3 |
| J2 | VIE 19/06 15:00 | EEUU vs Australia | Empate 1-1 | 36% | 27% | 37% | 1.4 - 1.4 |
| J2 | VIE 19/06 23:00 | **Turquía** vs Paraguay | Gana Turquía 1-0 | 38% | 29% | 33% | 1.4 - 1.3 |
| J3 | JUE 25/06 22:00 | Turquía vs EEUU | Empate 1-1 | 38% | 24% | 37% | 1.6 - 1.6 |
| J3 | JUE 25/06 22:00 | Paraguay vs Australia | Empate 1-1 | 34% | 29% | 37% | 1.2 - 1.2 |

P(1) = gana el primer equipo · P(X) = empate · P(2) = gana el segundo equipo.

### Tabla esperada del grupo

| Pos | Equipo | Pts esp. | GF-GC esp. | P(1°) | P(2°) | P(3° clasif.) | P(clasificar) |
|:-:|---|:-:|:-:|:-:|:-:|:-:|:-:|
| 1 | **Turquía** | 4.2 | 4.3-4.1 | 27% | 26% | 18% | **71%** |
| 2 | **EEUU** | 4.1 | 4.3-4.2 | 26% | 25% | 19% | **70%** |
| 3 | **Australia** | 4.1 | 3.9-3.9 | 25% | 25% | 19% | **69%** |
| 4 | **Paraguay** | 3.9 | 3.7-3.9 | 22% | 24% | 19% | **66%** |

---

## GRUPO E

### Partidos

| Jornada | Fecha | Partido | 🎯 Pronóstico | P(1) | P(X) | P(2) | Goles esperados |
|:-:|---|---|---|:-:|:-:|:-:|:-:|
| J1 | DOM 14/06 13:00 | **Alemania** vs Curazao | Gana Alemania 2-0 | 79% | 14% | 7% | 2.7 - 0.7 |
| J1 | DOM 14/06 19:00 | Costa de Marfil vs **Ecuador** | Gana Ecuador 0-1 | 20% | 30% | 50% | 0.8 - 1.4 |
| J2 | SÁB 20/06 16:00 | **Alemania** vs Costa de Marfil | Gana Alemania 1-0 | 54% | 25% | 21% | 1.8 - 1.0 |
| J2 | SÁB 20/06 20:00 | **Ecuador** vs Curazao | Gana Ecuador 2-0 | 77% | 16% | 7% | 2.5 - 0.6 |
| J3 | JUE 25/06 16:00 | Curazao vs **Costa de Marfil** | Gana Costa de Marfil 0-2 | 13% | 19% | 67% | 0.9 - 2.3 |
| J3 | JUE 25/06 16:00 | Ecuador vs Alemania | Empate 1-1 | 37% | 29% | 34% | 1.3 - 1.2 |

P(1) = gana el primer equipo · P(X) = empate · P(2) = gana el segundo equipo.

### Tabla esperada del grupo

| Pos | Equipo | Pts esp. | GF-GC esp. | P(1°) | P(2°) | P(3° clasif.) | P(clasificar) |
|:-:|---|:-:|:-:|:-:|:-:|:-:|:-:|
| 1 | **Alemania** | 5.7 | 5.7-3.0 | 43% | 35% | 16% | **93%** |
| 2 | **Ecuador** | 5.7 | 5.1-2.6 | 41% | 36% | 16% | **93%** |
| 3 | **Costa de Marfil** | 4.0 | 4.1-4.1 | 15% | 25% | 34% | **74%** |
| 4 | Curazao | 1.3 | 2.3-7.5 | 1% | 4% | 9% | **14%** |

---

## GRUPO F

### Partidos

| Jornada | Fecha | Partido | 🎯 Pronóstico | P(1) | P(X) | P(2) | Goles esperados |
|:-:|---|---|---|:-:|:-:|:-:|:-:|
| J1 | DOM 14/06 16:00 | Países Bajos vs Japón | Empate 1-1 | 36% | 30% | 34% | 1.2 - 1.1 |
| J1 | DOM 14/06 22:00 | Suecia vs Túnez | Empate 1-1 | 37% | 28% | 35% | 1.3 - 1.3 |
| J2 | SÁB 20/06 13:00 | **Países Bajos** vs Suecia | Gana Países Bajos 2-0 | 68% | 19% | 13% | 2.4 - 1.0 |
| J2 | DOM 21/06 00:00 | Túnez vs **Japón** | Gana Japón 0-1 | 15% | 24% | 61% | 0.8 - 1.8 |
| J3 | JUE 25/06 19:00 | **Japón** vs Suecia | Gana Japón 2-0 | 65% | 20% | 15% | 2.2 - 1.0 |
| J3 | JUE 25/06 19:00 | Túnez vs **Países Bajos** | Gana Países Bajos 0-2 | 14% | 22% | 64% | 0.9 - 2.1 |

P(1) = gana el primer equipo · P(X) = empate · P(2) = gana el segundo equipo.

### Tabla esperada del grupo

| Pos | Equipo | Pts esp. | GF-GC esp. | P(1°) | P(2°) | P(3° clasif.) | P(clasificar) |
|:-:|---|:-:|:-:|:-:|:-:|:-:|:-:|
| 1 | **Países Bajos** | 5.8 | 5.7-3.0 | 47% | 33% | 12% | **92%** |
| 2 | **Japón** | 5.5 | 5.2-2.9 | 41% | 37% | 13% | **90%** |
| 3 | Túnez | 2.7 | 2.9-5.2 | 6% | 15% | 22% | **43%** |
| 4 | Suecia | 2.6 | 3.3-6.0 | 6% | 15% | 21% | **42%** |

---

## GRUPO G

### Partidos

| Jornada | Fecha | Partido | 🎯 Pronóstico | P(1) | P(X) | P(2) | Goles esperados |
|:-:|---|---|---|:-:|:-:|:-:|:-:|
| J1 | LUN 15/06 15:00 | **Bélgica** vs Egipto | Gana Bélgica 1-0 | 56% | 26% | 18% | 1.7 - 0.8 |
| J1 | LUN 15/06 21:00 | **Irán** vs Nueva Zelanda | Gana Irán 1-0 | 55% | 26% | 19% | 1.7 - 0.9 |
| J2 | DOM 21/06 15:00 | **Bélgica** vs Irán | Gana Bélgica 2-1 | 45% | 26% | 29% | 1.6 - 1.3 |
| J2 | DOM 21/06 21:00 | Nueva Zelanda vs **Egipto** | Gana Egipto 0-1 | 29% | 30% | 41% | 1.1 - 1.3 |
| J3 | VIE 26/06 23:00 | Egipto vs **Irán** | Gana Irán 0-1 | 24% | 28% | 48% | 1.0 - 1.5 |
| J3 | VIE 26/06 23:00 | Nueva Zelanda vs **Bélgica** | Gana Bélgica 0-2 | 15% | 20% | 66% | 1.0 - 2.3 |

P(1) = gana el primer equipo · P(X) = empate · P(2) = gana el segundo equipo.

### Tabla esperada del grupo

| Pos | Equipo | Pts esp. | GF-GC esp. | P(1°) | P(2°) | P(3° clasif.) | P(clasificar) |
|:-:|---|:-:|:-:|:-:|:-:|:-:|:-:|
| 1 | **Bélgica** | 5.7 | 5.6-3.1 | 51% | 28% | 12% | **91%** |
| 2 | **Irán** | 4.8 | 4.4-3.5 | 30% | 33% | 17% | **81%** |
| 3 | **Egipto** | 3.3 | 3.1-4.2 | 11% | 23% | 22% | **56%** |
| 4 | Nueva Zelanda | 2.6 | 3.0-5.3 | 7% | 16% | 18% | **41%** |

---

## GRUPO H

### Partidos

| Jornada | Fecha | Partido | 🎯 Pronóstico | P(1) | P(X) | P(2) | Goles esperados |
|:-:|---|---|---|:-:|:-:|:-:|:-:|
| J1 | LUN 15/06 12:00 | **España** vs Cabo Verde | Gana España 2-0 | 86% | 10% | 3% | 3.0 - 0.5 |
| J1 | LUN 15/06 18:00 | Arabia S. vs **Uruguay** | Gana Uruguay 0-1 | 13% | 25% | 61% | 0.6 - 1.7 |
| J2 | DOM 21/06 12:00 | **España** vs Arabia S. | Gana España 2-0 | 83% | 12% | 5% | 2.9 - 0.6 |
| J2 | DOM 21/06 18:00 | **Uruguay** vs Cabo Verde | Gana Uruguay 2-0 | 70% | 20% | 10% | 2.1 - 0.6 |
| J3 | VIE 26/06 20:00 | Cabo Verde vs **Arabia S.** | Gana Arabia S. 1-2 | 29% | 27% | 44% | 1.2 - 1.6 |
| J3 | VIE 26/06 20:00 | Uruguay vs **España** | Gana España 0-1 | 17% | 25% | 57% | 0.8 - 1.7 |

P(1) = gana el primer equipo · P(X) = empate · P(2) = gana el segundo equipo.

### Tabla esperada del grupo

| Pos | Equipo | Pts esp. | GF-GC esp. | P(1°) | P(2°) | P(3° clasif.) | P(clasificar) |
|:-:|---|:-:|:-:|:-:|:-:|:-:|:-:|
| 1 | **España** | 7.3 | 7.6-1.9 | 76% | 20% | 3% | **99%** |
| 2 | **Uruguay** | 5.2 | 4.6-3.0 | 20% | 57% | 12% | **90%** |
| 3 | Arabia S. | 2.5 | 2.8-5.8 | 2% | 14% | 24% | **40%** |
| 4 | Cabo Verde | 1.8 | 2.4-6.7 | 1% | 8% | 15% | **25%** |

---

## GRUPO I

### Partidos

| Jornada | Fecha | Partido | 🎯 Pronóstico | P(1) | P(X) | P(2) | Goles esperados |
|:-:|---|---|---|:-:|:-:|:-:|:-:|
| J1 | MAR 16/06 15:00 | **Francia** vs Senegal | Gana Francia 1-0 | 56% | 26% | 19% | 1.7 - 0.9 |
| J1 | MAR 16/06 18:00 | Irak vs **Noruega** | Gana Noruega 0-2 | 11% | 22% | 67% | 0.7 - 1.9 |
| J2 | LUN 22/06 17:00 | **Francia** vs Irak | Gana Francia 2-0 | 76% | 16% | 8% | 2.4 - 0.6 |
| J2 | LUN 22/06 20:00 | **Noruega** vs Senegal | Gana Noruega 1-0 | 45% | 27% | 28% | 1.5 - 1.2 |
| J3 | VIE 26/06 15:00 | Noruega vs **Francia** | Gana Francia 1-2 | 26% | 26% | 48% | 1.2 - 1.7 |
| J3 | VIE 26/06 15:00 | **Senegal** vs Irak | Gana Senegal 2-0 | 61% | 24% | 16% | 1.9 - 0.9 |

P(1) = gana el primer equipo · P(X) = empate · P(2) = gana el segundo equipo.

### Tabla esperada del grupo

| Pos | Equipo | Pts esp. | GF-GC esp. | P(1°) | P(2°) | P(3° clasif.) | P(clasificar) |
|:-:|---|:-:|:-:|:-:|:-:|:-:|:-:|
| 1 | **Francia** | 6.1 | 5.9-2.7 | 55% | 29% | 11% | **95%** |
| 2 | **Noruega** | 4.9 | 4.7-3.6 | 27% | 36% | 20% | **84%** |
| 3 | **Senegal** | 4.0 | 3.9-4.1 | 16% | 28% | 28% | **72%** |
| 4 | Irak | 1.7 | 2.2-6.3 | 2% | 7% | 12% | **21%** |

---

## GRUPO J

### Partidos

| Jornada | Fecha | Partido | 🎯 Pronóstico | P(1) | P(X) | P(2) | Goles esperados |
|:-:|---|---|---|:-:|:-:|:-:|:-:|
| J1 | MAR 16/06 21:00 | **Argentina** vs Argelia | Gana Argentina 1-0 | 59% | 25% | 17% | 1.8 - 0.8 |
| J1 | MIÉ 17/06 00:00 | **Austria** vs Jordania | Gana Austria 1-0 | 49% | 27% | 24% | 1.6 - 1.1 |
| J2 | LUN 22/06 13:00 | **Argentina** vs Austria | Gana Argentina 2-0 | 65% | 22% | 13% | 2.0 - 0.8 |
| J2 | LUN 22/06 23:00 | Jordania vs **Argelia** | Gana Argelia 1-2 | 21% | 23% | 56% | 1.1 - 2.0 |
| J3 | SÁB 27/06 22:00 | **Argelia** vs Austria | Gana Argelia 2-1 | 41% | 28% | 31% | 1.5 - 1.2 |
| J3 | SÁB 27/06 22:00 | Jordania vs **Argentina** | Gana Argentina 0-2 | 8% | 15% | 77% | 0.8 - 2.8 |

P(1) = gana el primer equipo · P(X) = empate · P(2) = gana el segundo equipo.

### Tabla esperada del grupo

| Pos | Equipo | Pts esp. | GF-GC esp. | P(1°) | P(2°) | P(3° clasif.) | P(clasificar) |
|:-:|---|:-:|:-:|:-:|:-:|:-:|:-:|
| 1 | **Argentina** | 6.6 | 6.6-2.5 | 69% | 21% | 7% | **97%** |
| 2 | **Argelia** | 4.2 | 4.3-4.2 | 17% | 37% | 21% | **74%** |
| 3 | **Austria** | 3.6 | 3.7-4.5 | 11% | 28% | 24% | **63%** |
| 4 | Jordania | 2.2 | 3.0-6.4 | 4% | 14% | 15% | **33%** |

---

## GRUPO K

### Partidos

| Jornada | Fecha | Partido | 🎯 Pronóstico | P(1) | P(X) | P(2) | Goles esperados |
|:-:|---|---|---|:-:|:-:|:-:|:-:|
| J1 | MIÉ 17/06 13:00 | **Portugal** vs RD Congo | Gana Portugal 1-0 | 63% | 24% | 12% | 1.8 - 0.6 |
| J1 | MIÉ 17/06 22:00 | Uzbekistán vs **Colombia** | Gana Colombia 0-1 | 14% | 24% | 62% | 0.7 - 1.8 |
| J2 | MAR 23/06 13:00 | **Portugal** vs Uzbekistán | Gana Portugal 1-0 | 60% | 24% | 16% | 1.8 - 0.8 |
| J2 | MAR 23/06 22:00 | **Colombia** vs RD Congo | Gana Colombia 2-0 | 68% | 21% | 11% | 2.0 - 0.7 |
| J3 | SÁB 27/06 19:30 | **Colombia** vs Portugal | Gana Colombia 2-1 | 40% | 27% | 34% | 1.5 - 1.4 |
| J3 | SÁB 27/06 19:30 | RD Congo vs **Uzbekistán** | Gana Uzbekistán 0-1 | 29% | 30% | 41% | 1.0 - 1.2 |

P(1) = gana el primer equipo · P(X) = empate · P(2) = gana el segundo equipo.

### Tabla esperada del grupo

| Pos | Equipo | Pts esp. | GF-GC esp. | P(1°) | P(2°) | P(3° clasif.) | P(clasificar) |
|:-:|---|:-:|:-:|:-:|:-:|:-:|:-:|
| 1 | **Colombia** | 5.8 | 5.3-2.8 | 48% | 33% | 11% | **92%** |
| 2 | **Portugal** | 5.5 | 4.9-3.0 | 40% | 37% | 13% | **90%** |
| 3 | Uzbekistán | 2.9 | 2.8-4.7 | 7% | 17% | 23% | **48%** |
| 4 | RD Congo | 2.3 | 2.3-5.0 | 4% | 12% | 19% | **35%** |

---

## GRUPO L

### Partidos

| Jornada | Fecha | Partido | 🎯 Pronóstico | P(1) | P(X) | P(2) | Goles esperados |
|:-:|---|---|---|:-:|:-:|:-:|:-:|
| J1 | MIÉ 17/06 16:00 | **Inglaterra** vs Croacia | Gana Inglaterra 1-0 | 51% | 27% | 21% | 1.6 - 0.9 |
| J1 | MIÉ 17/06 19:00 | Ghana vs **Panamá** | Gana Panamá 0-1 | 21% | 25% | 53% | 1.0 - 1.7 |
| J2 | MAR 23/06 16:00 | **Inglaterra** vs Ghana | Gana Inglaterra 2-0 | 80% | 14% | 6% | 2.7 - 0.6 |
| J2 | MAR 23/06 19:00 | Panamá vs **Croacia** | Gana Croacia 1-2 | 20% | 23% | 57% | 1.1 - 2.0 |
| J3 | SÁB 27/06 17:00 | Panamá vs **Inglaterra** | Gana Inglaterra 0-2 | 12% | 18% | 70% | 0.9 - 2.4 |
| J3 | SÁB 27/06 17:00 | **Croacia** vs Ghana | Gana Croacia 2-0 | 70% | 19% | 11% | 2.4 - 0.8 |

P(1) = gana el primer equipo · P(X) = empate · P(2) = gana el segundo equipo.

### Tabla esperada del grupo

| Pos | Equipo | Pts esp. | GF-GC esp. | P(1°) | P(2°) | P(3° clasif.) | P(clasificar) |
|:-:|---|:-:|:-:|:-:|:-:|:-:|:-:|
| 1 | **Inglaterra** | 6.7 | 6.6-2.4 | 64% | 26% | 7% | **97%** |
| 2 | **Croacia** | 5.1 | 5.3-3.5 | 27% | 45% | 17% | **89%** |
| 3 | **Panamá** | 3.2 | 3.7-5.4 | 7% | 21% | 28% | **56%** |
| 4 | Ghana | 1.7 | 2.5-6.8 | 2% | 8% | 13% | **23%** |
