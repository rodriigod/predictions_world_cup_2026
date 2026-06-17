# 🏆 Mundial 2026 — Predicción de la Fase de Grupos

**Método:** Regresión de Poisson (goles esperados λ por equipo) + corrección Dixon-Coles + Simulación de Monte Carlo: **50,000 iteraciones del torneo completo** — cada uno de los 72 partidos se simula 50,000 veces con perturbación estocástica de las condiciones del día (clima, estado físico, campo) e incentivos dinámicos en la jornada 3 (rotaciones de clasificados, empates que sirven a ambos, urgencias).

**Clasifican a 16avos:** los 2 primeros de cada grupo + los 8 mejores terceros de los 12 grupos.

El 🎯 Pronóstico es el resultado 1X2 más probable con su marcador más frecuente (útil para llenar la polla). Las probabilidades muestran cuán confiable es cada pronóstico.

## GRUPO A

### Partidos

| Jornada | Fecha | Partido | 🎯 Pronóstico | P(1) | P(X) | P(2) | Goles esperados |
|:-:|---|---|---|:-:|:-:|:-:|:-:|
| J1 | JUE 11/06 15:00 | **México** vs Sudáfrica | Gana México 2-0 | 77% | 16% | 6% | 2.3 - 0.5 |
| J1 | JUE 11/06 22:00 | **Corea del Sur** vs Rep. Checa | Gana Corea del Sur 1-0 | 45% | 27% | 27% | 1.6 - 1.1 |
| J2 | JUE 18/06 12:00 | **Rep. Checa** vs Sudáfrica | Gana Rep. Checa 2-1 | 53% | 25% | 22% | 1.8 - 1.1 |
| J2 | JUE 18/06 21:00 | **México** vs Corea del Sur | Gana México 1-0 | 60% | 24% | 16% | 1.8 - 0.8 |
| J3 | MIÉ 24/06 21:00 | Rep. Checa vs **México** | Gana México 0-2 | 12% | 20% | 68% | 0.8 - 2.2 |
| J3 | MIÉ 24/06 21:00 | Sudáfrica vs **Corea del Sur** | Gana Corea del Sur 0-2 | 16% | 22% | 62% | 1.0 - 2.1 |

P(1) = gana el primer equipo · P(X) = empate · P(2) = gana el segundo equipo.

### Tabla esperada del grupo

| Pos | Equipo | Pts esp. | GF-GC esp. | P(1°) | P(2°) | P(3° clasif.) | P(clasificar) |
|:-:|---|:-:|:-:|:-:|:-:|:-:|:-:|
| 1 | **México** | 6.8 | 6.3-2.1 | 70% | 22% | 6% | **97%** |
| 2 | **Corea del Sur** | 4.4 | 4.4-3.9 | 18% | 40% | 21% | **79%** |
| 3 | **Rep. Checa** | 3.5 | 3.7-4.8 | 9% | 28% | 25% | **62%** |
| 4 | Sudáfrica | 2.0 | 2.6-6.2 | 3% | 11% | 14% | **28%** |

---

## GRUPO B

### Partidos

| Jornada | Fecha | Partido | 🎯 Pronóstico | P(1) | P(X) | P(2) | Goles esperados |
|:-:|---|---|---|:-:|:-:|:-:|:-:|
| J1 | VIE 12/06 15:00 | **Canadá** vs Bosnia y Her. | Gana Canadá 2-0 | 70% | 22% | 9% | 1.9 - 0.5 |
| J1 | SÁB 13/06 15:00 | Catar vs **Suiza** | Gana Suiza 0-2 | 4% | 10% | 86% | 0.6 - 3.1 |
| J2 | JUE 18/06 15:00 | **Suiza** vs Bosnia y Her. | Gana Suiza 2-0 | 76% | 16% | 8% | 2.5 - 0.7 |
| J2 | JUE 18/06 18:00 | **Canadá** vs Catar | Gana Canadá 2-0 | 84% | 12% | 4% | 2.8 - 0.5 |
| J3 | MIÉ 24/06 15:00 | Suiza vs Canadá | Empate 1-1 | 34% | 30% | 36% | 1.2 - 1.2 |
| J3 | MIÉ 24/06 15:00 | **Bosnia y Her.** vs Catar | Gana Bosnia y Her. 2-1 | 55% | 24% | 22% | 1.9 - 1.2 |

P(1) = gana el primer equipo · P(X) = empate · P(2) = gana el segundo equipo.

### Tabla esperada del grupo

| Pos | Equipo | Pts esp. | GF-GC esp. | P(1°) | P(2°) | P(3° clasif.) | P(clasificar) |
|:-:|---|:-:|:-:|:-:|:-:|:-:|:-:|
| 1 | **Suiza** | 6.4 | 6.7-2.5 | 50% | 41% | 7% | **98%** |
| 2 | **Canadá** | 6.3 | 5.9-2.2 | 46% | 44% | 7% | **97%** |
| 3 | Bosnia y Her. | 2.7 | 3.2-5.6 | 3% | 12% | 33% | **47%** |
| 4 | Catar | 1.4 | 2.2-7.7 | 1% | 4% | 11% | **15%** |

---

## GRUPO C

### Partidos

| Jornada | Fecha | Partido | 🎯 Pronóstico | P(1) | P(X) | P(2) | Goles esperados |
|:-:|---|---|---|:-:|:-:|:-:|:-:|
| J1 | SÁB 13/06 18:00 | **Brasil** vs Marruecos | Gana Brasil 1-0 | 42% | 29% | 29% | 1.4 - 1.1 |
| J1 | SÁB 13/06 21:00 | Haití vs **Escocia** | Gana Escocia 0-1 | 22% | 26% | 53% | 1.0 - 1.7 |
| J2 | VIE 19/06 18:00 | Escocia vs **Marruecos** | Gana Marruecos 0-1 | 21% | 26% | 54% | 1.0 - 1.7 |
| J2 | VIE 19/06 20:30 | **Brasil** vs Haití | Gana Brasil 2-0 | 77% | 15% | 8% | 2.8 - 0.8 |
| J3 | MIÉ 24/06 18:00 | Escocia vs **Brasil** | Gana Brasil 1-2 | 17% | 21% | 63% | 1.1 - 2.2 |
| J3 | MIÉ 24/06 18:00 | **Marruecos** vs Haití | Gana Marruecos 2-0 | 69% | 19% | 12% | 2.3 - 0.8 |

P(1) = gana el primer equipo · P(X) = empate · P(2) = gana el segundo equipo.

### Tabla esperada del grupo

| Pos | Equipo | Pts esp. | GF-GC esp. | P(1°) | P(2°) | P(3° clasif.) | P(clasificar) |
|:-:|---|:-:|:-:|:-:|:-:|:-:|:-:|
| 1 | **Brasil** | 6.1 | 6.4-3.0 | 54% | 30% | 10% | **95%** |
| 2 | **Marruecos** | 5.3 | 5.1-3.2 | 33% | 39% | 17% | **89%** |
| 3 | **Escocia** | 3.4 | 3.8-5.0 | 10% | 22% | 28% | **60%** |
| 4 | Haití | 1.9 | 2.7-6.8 | 3% | 9% | 13% | **25%** |

---

## GRUPO D

### Partidos

| Jornada | Fecha | Partido | 🎯 Pronóstico | P(1) | P(X) | P(2) | Goles esperados |
|:-:|---|---|---|:-:|:-:|:-:|:-:|
| J1 | VIE 12/06 21:00 | **EEUU** vs Paraguay | Gana EEUU 1-0 | 38% | 29% | 33% | 1.3 - 1.2 |
| J1 | DOM 14/06 00:00 | Australia vs Turquía | Empate 1-1 | 36% | 29% | 35% | 1.3 - 1.2 |
| J2 | VIE 19/06 15:00 | EEUU vs Australia | Empate 1-1 | 35% | 27% | 38% | 1.3 - 1.4 |
| J2 | VIE 19/06 23:00 | **Turquía** vs Paraguay | Gana Turquía 1-0 | 40% | 28% | 32% | 1.4 - 1.2 |
| J3 | JUE 25/06 22:00 | Turquía vs EEUU | Empate 1-1 | 37% | 25% | 38% | 1.6 - 1.6 |
| J3 | JUE 25/06 22:00 | Paraguay vs **Australia** | Gana Australia 0-1 | 32% | 29% | 39% | 1.2 - 1.3 |

P(1) = gana el primer equipo · P(X) = empate · P(2) = gana el segundo equipo.

### Tabla esperada del grupo

| Pos | Equipo | Pts esp. | GF-GC esp. | P(1°) | P(2°) | P(3° clasif.) | P(clasificar) |
|:-:|---|:-:|:-:|:-:|:-:|:-:|:-:|
| 1 | **Australia** | 4.2 | 4.0-3.8 | 27% | 26% | 19% | **71%** |
| 2 | **Turquía** | 4.2 | 4.3-4.2 | 27% | 26% | 19% | **71%** |
| 3 | **EEUU** | 4.1 | 4.3-4.2 | 26% | 25% | 18% | **70%** |
| 4 | **Paraguay** | 3.8 | 3.6-4.0 | 20% | 23% | 20% | **64%** |

---

## GRUPO E

### Partidos

| Jornada | Fecha | Partido | 🎯 Pronóstico | P(1) | P(X) | P(2) | Goles esperados |
|:-:|---|---|---|:-:|:-:|:-:|:-:|
| J1 | DOM 14/06 13:00 | **Alemania** vs Curazao | Gana Alemania 2-0 | 79% | 14% | 7% | 2.7 - 0.7 |
| J1 | DOM 14/06 19:00 | Costa de Marfil vs **Ecuador** | Gana Ecuador 0-1 | 20% | 30% | 49% | 0.8 - 1.3 |
| J2 | SÁB 20/06 16:00 | **Alemania** vs Costa de Marfil | Gana Alemania 1-0 | 55% | 25% | 20% | 1.8 - 1.0 |
| J2 | SÁB 20/06 20:00 | **Ecuador** vs Curazao | Gana Ecuador 2-0 | 76% | 16% | 7% | 2.4 - 0.6 |
| J3 | JUE 25/06 16:00 | Curazao vs **Costa de Marfil** | Gana Costa de Marfil 0-2 | 14% | 19% | 67% | 1.0 - 2.3 |
| J3 | JUE 25/06 16:00 | Ecuador vs Alemania | Empate 1-1 | 36% | 29% | 36% | 1.2 - 1.2 |

P(1) = gana el primer equipo · P(X) = empate · P(2) = gana el segundo equipo.

### Tabla esperada del grupo

| Pos | Equipo | Pts esp. | GF-GC esp. | P(1°) | P(2°) | P(3° clasif.) | P(clasificar) |
|:-:|---|:-:|:-:|:-:|:-:|:-:|:-:|
| 1 | **Alemania** | 5.8 | 5.8-2.9 | 45% | 33% | 15% | **93%** |
| 2 | **Ecuador** | 5.6 | 5.0-2.6 | 39% | 36% | 16% | **92%** |
| 3 | **Costa de Marfil** | 4.0 | 4.1-4.1 | 14% | 26% | 33% | **73%** |
| 4 | Curazao | 1.3 | 2.3-7.4 | 1% | 5% | 9% | **15%** |

---

## GRUPO F

### Partidos

| Jornada | Fecha | Partido | 🎯 Pronóstico | P(1) | P(X) | P(2) | Goles esperados |
|:-:|---|---|---|:-:|:-:|:-:|:-:|
| J1 | DOM 14/06 16:00 | **Países Bajos** vs Japón | Gana Países Bajos 1-0 | 37% | 31% | 32% | 1.2 - 1.1 |
| J1 | DOM 14/06 22:00 | Suecia vs Túnez | Empate 1-1 | 36% | 28% | 35% | 1.3 - 1.3 |
| J2 | SÁB 20/06 13:00 | **Países Bajos** vs Suecia | Gana Países Bajos 2-0 | 69% | 18% | 13% | 2.5 - 1.0 |
| J2 | DOM 21/06 00:00 | Túnez vs **Japón** | Gana Japón 0-1 | 15% | 25% | 60% | 0.8 - 1.8 |
| J3 | JUE 25/06 19:00 | **Japón** vs Suecia | Gana Japón 2-0 | 64% | 21% | 15% | 2.2 - 1.0 |
| J3 | JUE 25/06 19:00 | Túnez vs **Países Bajos** | Gana Países Bajos 0-2 | 15% | 22% | 64% | 0.9 - 2.0 |

P(1) = gana el primer equipo · P(X) = empate · P(2) = gana el segundo equipo.

### Tabla esperada del grupo

| Pos | Equipo | Pts esp. | GF-GC esp. | P(1°) | P(2°) | P(3° clasif.) | P(clasificar) |
|:-:|---|:-:|:-:|:-:|:-:|:-:|:-:|
| 1 | **Países Bajos** | 5.8 | 5.7-3.0 | 48% | 33% | 12% | **92%** |
| 2 | **Japón** | 5.5 | 5.1-2.9 | 40% | 37% | 13% | **90%** |
| 3 | Túnez | 2.7 | 3.0-5.1 | 6% | 16% | 22% | **44%** |
| 4 | Suecia | 2.6 | 3.3-6.0 | 6% | 15% | 20% | **41%** |

---

## GRUPO G

### Partidos

| Jornada | Fecha | Partido | 🎯 Pronóstico | P(1) | P(X) | P(2) | Goles esperados |
|:-:|---|---|---|:-:|:-:|:-:|:-:|
| J1 | LUN 15/06 15:00 | **Bélgica** vs Egipto | Gana Bélgica 1-0 | 56% | 26% | 18% | 1.7 - 0.8 |
| J1 | LUN 15/06 21:00 | **Irán** vs Nueva Zelanda | Gana Irán 1-0 | 54% | 26% | 20% | 1.7 - 0.9 |
| J2 | DOM 21/06 15:00 | **Bélgica** vs Irán | Gana Bélgica 2-1 | 46% | 26% | 28% | 1.6 - 1.2 |
| J2 | DOM 21/06 21:00 | Nueva Zelanda vs **Egipto** | Gana Egipto 0-1 | 29% | 29% | 41% | 1.1 - 1.3 |
| J3 | VIE 26/06 23:00 | Egipto vs **Irán** | Gana Irán 0-1 | 24% | 28% | 47% | 1.0 - 1.5 |
| J3 | VIE 26/06 23:00 | Nueva Zelanda vs **Bélgica** | Gana Bélgica 1-2 | 15% | 20% | 65% | 1.0 - 2.3 |

P(1) = gana el primer equipo · P(X) = empate · P(2) = gana el segundo equipo.

### Tabla esperada del grupo

| Pos | Equipo | Pts esp. | GF-GC esp. | P(1°) | P(2°) | P(3° clasif.) | P(clasificar) |
|:-:|---|:-:|:-:|:-:|:-:|:-:|:-:|
| 1 | **Bélgica** | 5.7 | 5.6-3.1 | 52% | 27% | 12% | **91%** |
| 2 | **Irán** | 4.7 | 4.4-3.5 | 29% | 34% | 18% | **80%** |
| 3 | **Egipto** | 3.3 | 3.1-4.2 | 12% | 23% | 22% | **57%** |
| 4 | Nueva Zelanda | 2.7 | 3.0-5.3 | 8% | 16% | 18% | **42%** |

---

## GRUPO H

### Partidos

| Jornada | Fecha | Partido | 🎯 Pronóstico | P(1) | P(X) | P(2) | Goles esperados |
|:-:|---|---|---|:-:|:-:|:-:|:-:|
| J1 | LUN 15/06 12:00 | **España** vs Cabo Verde | Gana España 2-0 | 86% | 10% | 3% | 3.0 - 0.5 |
| J1 | LUN 15/06 18:00 | Arabia S. vs **Uruguay** | Gana Uruguay 0-1 | 14% | 26% | 61% | 0.6 - 1.6 |
| J2 | DOM 21/06 12:00 | **España** vs Arabia S. | Gana España 2-0 | 82% | 13% | 5% | 2.8 - 0.6 |
| J2 | DOM 21/06 18:00 | **Uruguay** vs Cabo Verde | Gana Uruguay 2-0 | 70% | 20% | 10% | 2.1 - 0.6 |
| J3 | VIE 26/06 20:00 | Cabo Verde vs **Arabia S.** | Gana Arabia S. 1-2 | 28% | 26% | 45% | 1.2 - 1.6 |
| J3 | VIE 26/06 20:00 | Uruguay vs **España** | Gana España 0-1 | 17% | 25% | 58% | 0.8 - 1.7 |

P(1) = gana el primer equipo · P(X) = empate · P(2) = gana el segundo equipo.

### Tabla esperada del grupo

| Pos | Equipo | Pts esp. | GF-GC esp. | P(1°) | P(2°) | P(3° clasif.) | P(clasificar) |
|:-:|---|:-:|:-:|:-:|:-:|:-:|:-:|
| 1 | **España** | 7.3 | 7.6-1.9 | 77% | 20% | 3% | **99%** |
| 2 | **Uruguay** | 5.2 | 4.6-3.0 | 20% | 57% | 12% | **89%** |
| 3 | Arabia S. | 2.6 | 2.8-5.7 | 2% | 15% | 25% | **42%** |
| 4 | Cabo Verde | 1.8 | 2.4-6.7 | 1% | 8% | 15% | **24%** |

---

## GRUPO I

### Partidos

| Jornada | Fecha | Partido | 🎯 Pronóstico | P(1) | P(X) | P(2) | Goles esperados |
|:-:|---|---|---|:-:|:-:|:-:|:-:|
| J1 | MAR 16/06 15:00 | **Francia** vs Senegal | Gana Francia 1-0 | 54% | 26% | 20% | 1.7 - 0.9 |
| J1 | MAR 16/06 18:00 | Irak vs **Noruega** | Gana Noruega 0-2 | 11% | 22% | 67% | 0.7 - 2.0 |
| J2 | LUN 22/06 17:00 | **Francia** vs Irak | Gana Francia 2-0 | 76% | 16% | 7% | 2.4 - 0.6 |
| J2 | LUN 22/06 20:00 | **Noruega** vs Senegal | Gana Noruega 1-0 | 46% | 27% | 28% | 1.5 - 1.1 |
| J3 | VIE 26/06 15:00 | Noruega vs **Francia** | Gana Francia 1-2 | 27% | 26% | 47% | 1.2 - 1.7 |
| J3 | VIE 26/06 15:00 | **Senegal** vs Irak | Gana Senegal 2-0 | 61% | 23% | 15% | 1.9 - 0.9 |

P(1) = gana el primer equipo · P(X) = empate · P(2) = gana el segundo equipo.

### Tabla esperada del grupo

| Pos | Equipo | Pts esp. | GF-GC esp. | P(1°) | P(2°) | P(3° clasif.) | P(clasificar) |
|:-:|---|:-:|:-:|:-:|:-:|:-:|:-:|
| 1 | **Francia** | 6.0 | 5.8-2.8 | 52% | 29% | 12% | **94%** |
| 2 | **Noruega** | 5.0 | 4.7-3.5 | 29% | 36% | 20% | **85%** |
| 3 | **Senegal** | 4.0 | 4.0-4.1 | 16% | 28% | 28% | **72%** |
| 4 | Irak | 1.6 | 2.1-6.3 | 2% | 7% | 11% | **20%** |

---

## GRUPO J

### Partidos

| Jornada | Fecha | Partido | 🎯 Pronóstico | P(1) | P(X) | P(2) | Goles esperados |
|:-:|---|---|---|:-:|:-:|:-:|:-:|
| J1 | MAR 16/06 21:00 | **Argentina** vs Argelia | Gana Argentina 1-0 | 56% | 26% | 18% | 1.7 - 0.9 |
| J1 | MIÉ 17/06 00:00 | **Austria** vs Jordania | Gana Austria 1-0 | 45% | 27% | 27% | 1.5 - 1.1 |
| J2 | LUN 22/06 13:00 | **Argentina** vs Austria | Gana Argentina 2-0 | 64% | 22% | 14% | 2.0 - 0.8 |
| J2 | LUN 22/06 23:00 | Jordania vs **Argelia** | Gana Argelia 1-2 | 22% | 24% | 54% | 1.2 - 1.9 |
| J3 | SÁB 27/06 22:00 | **Argelia** vs Austria | Gana Argelia 1-0 | 43% | 28% | 30% | 1.5 - 1.2 |
| J3 | SÁB 27/06 22:00 | Jordania vs **Argentina** | Gana Argentina 0-2 | 10% | 17% | 73% | 0.9 - 2.6 |

P(1) = gana el primer equipo · P(X) = empate · P(2) = gana el segundo equipo.

### Tabla esperada del grupo

| Pos | Equipo | Pts esp. | GF-GC esp. | P(1°) | P(2°) | P(3° clasif.) | P(clasificar) |
|:-:|---|:-:|:-:|:-:|:-:|:-:|:-:|
| 1 | **Argentina** | 6.5 | 6.3-2.6 | 66% | 22% | 8% | **96%** |
| 2 | **Argelia** | 4.2 | 4.3-4.1 | 18% | 36% | 20% | **74%** |
| 3 | **Austria** | 3.4 | 3.5-4.6 | 11% | 26% | 23% | **60%** |
| 4 | Jordania | 2.5 | 3.2-6.1 | 5% | 16% | 17% | **38%** |

---

## GRUPO K

### Partidos

| Jornada | Fecha | Partido | 🎯 Pronóstico | P(1) | P(X) | P(2) | Goles esperados |
|:-:|---|---|---|:-:|:-:|:-:|:-:|
| J1 | MIÉ 17/06 13:00 | **Portugal** vs RD Congo | Gana Portugal 1-0 | 62% | 25% | 13% | 1.7 - 0.7 |
| J1 | MIÉ 17/06 22:00 | Uzbekistán vs **Colombia** | Gana Colombia 0-1 | 15% | 25% | 60% | 0.7 - 1.7 |
| J2 | MAR 23/06 13:00 | **Portugal** vs Uzbekistán | Gana Portugal 1-0 | 58% | 25% | 17% | 1.8 - 0.9 |
| J2 | MAR 23/06 22:00 | **Colombia** vs RD Congo | Gana Colombia 2-0 | 66% | 22% | 12% | 2.0 - 0.7 |
| J3 | SÁB 27/06 19:30 | **Colombia** vs Portugal | Gana Colombia 2-1 | 40% | 27% | 33% | 1.5 - 1.4 |
| J3 | SÁB 27/06 19:30 | RD Congo vs **Uzbekistán** | Gana Uzbekistán 0-1 | 28% | 31% | 41% | 1.0 - 1.2 |

P(1) = gana el primer equipo · P(X) = empate · P(2) = gana el segundo equipo.

### Tabla esperada del grupo

| Pos | Equipo | Pts esp. | GF-GC esp. | P(1°) | P(2°) | P(3° clasif.) | P(clasificar) |
|:-:|---|:-:|:-:|:-:|:-:|:-:|:-:|
| 1 | **Colombia** | 5.7 | 5.2-2.8 | 48% | 32% | 11% | **92%** |
| 2 | **Portugal** | 5.4 | 4.8-3.0 | 39% | 36% | 13% | **88%** |
| 3 | **Uzbekistán** | 3.0 | 2.9-4.5 | 8% | 19% | 24% | **51%** |
| 4 | RD Congo | 2.4 | 2.3-4.9 | 5% | 13% | 18% | **35%** |

---

## GRUPO L

### Partidos

| Jornada | Fecha | Partido | 🎯 Pronóstico | P(1) | P(X) | P(2) | Goles esperados |
|:-:|---|---|---|:-:|:-:|:-:|:-:|
| J1 | MIÉ 17/06 16:00 | **Inglaterra** vs Croacia | Gana Inglaterra 1-0 | 51% | 28% | 22% | 1.5 - 0.9 |
| J1 | MIÉ 17/06 19:00 | Ghana vs **Panamá** | Gana Panamá 0-1 | 22% | 26% | 52% | 1.0 - 1.7 |
| J2 | MAR 23/06 16:00 | **Inglaterra** vs Ghana | Gana Inglaterra 2-0 | 79% | 15% | 6% | 2.6 - 0.6 |
| J2 | MAR 23/06 19:00 | Panamá vs **Croacia** | Gana Croacia 1-2 | 20% | 23% | 57% | 1.1 - 2.0 |
| J3 | SÁB 27/06 17:00 | Panamá vs **Inglaterra** | Gana Inglaterra 0-2 | 11% | 18% | 71% | 0.9 - 2.4 |
| J3 | SÁB 27/06 17:00 | **Croacia** vs Ghana | Gana Croacia 2-0 | 69% | 19% | 12% | 2.3 - 0.9 |

P(1) = gana el primer equipo · P(X) = empate · P(2) = gana el segundo equipo.

### Tabla esperada del grupo

| Pos | Equipo | Pts esp. | GF-GC esp. | P(1°) | P(2°) | P(3° clasif.) | P(clasificar) |
|:-:|---|:-:|:-:|:-:|:-:|:-:|:-:|
| 1 | **Inglaterra** | 6.6 | 6.6-2.4 | 64% | 26% | 7% | **97%** |
| 2 | **Croacia** | 5.1 | 5.2-3.5 | 27% | 44% | 16% | **88%** |
| 3 | **Panamá** | 3.2 | 3.7-5.5 | 7% | 21% | 27% | **55%** |
| 4 | Ghana | 1.8 | 2.5-6.6 | 2% | 8% | 14% | **24%** |
