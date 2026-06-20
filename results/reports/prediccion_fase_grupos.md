# 🏆 Mundial 2026 — Predicción de la Fase de Grupos

**Método:** Regresión de Poisson (goles esperados λ por equipo) + corrección Dixon-Coles + Simulación de Monte Carlo: **50,000 iteraciones del torneo completo** — cada uno de los 72 partidos se simula 50,000 veces con perturbación estocástica de las condiciones del día (clima, estado físico, campo) e incentivos dinámicos en la jornada 3 (rotaciones de clasificados, empates que sirven a ambos, urgencias).

**Clasifican a 16avos:** los 2 primeros de cada grupo + los 8 mejores terceros de los 12 grupos.

El 🎯 Pronóstico es el marcador que MAXIMIZA los puntos esperados de la polla (E[pts]=3·P(resultado)+2·P(marcador)): prioriza acertar el 1X2 y afina el marcador dentro de ese resultado. Las probabilidades muestran cuán confiable es cada pronóstico.

## GRUPO A

### Partidos

| Jornada | Fecha | Partido | 🎯 Pronóstico | P(1) | P(X) | P(2) | Goles esperados |
|:-:|---|---|---|:-:|:-:|:-:|:-:|
| J1 | JUE 11/06 15:00 | **México** vs Sudáfrica | Gana México 2-0 | 78% | 16% | 6% | 2.32 - 0.51 |
| J1 | JUE 11/06 22:00 | **Corea del Sur** vs Rep. Checa | Gana Corea del Sur 2-1 | 44% | 27% | 29% | 1.52 - 1.17 |
| J2 | JUE 18/06 12:00 | **Rep. Checa** vs Sudáfrica | Gana Rep. Checa 1-0 | 54% | 25% | 21% | 1.75 - 0.99 |
| J2 | JUE 18/06 21:00 | **México** vs Corea del Sur | Gana México 1-0 | 51% | 28% | 22% | 1.53 - 0.91 |
| J3 | MIÉ 24/06 21:00 | Rep. Checa vs **México** | Gana México 0-1 | 19% | 25% | 56% | 0.91 - 1.72 |
| J3 | MIÉ 24/06 21:00 | Sudáfrica vs **Corea del Sur** | Gana Corea del Sur 0-2 | 16% | 22% | 62% | 0.95 - 2.03 |

P(1) = gana el primer equipo · P(X) = empate · P(2) = gana el segundo equipo.

### Tabla esperada del grupo

| Pos | Equipo | Pts esp. | GF-GC esp. | P(1°) | P(2°) | P(3° clasif.) | P(clasificar) |
|:-:|---|:-:|:-:|:-:|:-:|:-:|:-:|
| 1 | **México** | 6.2 | 5.6-2.3 | 58% | 27% | 10% | **95%** |
| 2 | **Corea del Sur** | 4.6 | 4.5-3.7 | 24% | 35% | 21% | **80%** |
| 3 | **Rep. Checa** | 3.8 | 3.8-4.2 | 15% | 28% | 25% | **68%** |
| 4 | Sudáfrica | 1.9 | 2.4-6.1 | 3% | 10% | 14% | **26%** |

---

## GRUPO B

### Partidos

| Jornada | Fecha | Partido | 🎯 Pronóstico | P(1) | P(X) | P(2) | Goles esperados |
|:-:|---|---|---|:-:|:-:|:-:|:-:|
| J1 | VIE 12/06 15:00 | **Canadá** vs Bosnia y Her. | Gana Canadá 1-0 | 70% | 21% | 9% | 1.93 - 0.53 |
| J1 | SÁB 13/06 15:00 | Catar vs **Suiza** | Gana Suiza 0-3 | 4% | 11% | 85% | 0.57 - 3.00 |
| J2 | JUE 18/06 15:00 | **Suiza** vs Bosnia y Her. | Gana Suiza 2-0 | 67% | 21% | 12% | 2.08 - 0.78 |
| J2 | JUE 18/06 18:00 | **Canadá** vs Catar | Gana Canadá 2-0 | 79% | 15% | 6% | 2.58 - 0.60 |
| J3 | MIÉ 24/06 15:00 | **Suiza** vs Canadá | Gana Suiza 1-0 | 41% | 29% | 30% | 1.34 - 1.09 |
| J3 | MIÉ 24/06 15:00 | **Bosnia y Her.** vs Catar | Gana Bosnia y Her. 2-0 | 60% | 23% | 17% | 2.03 - 0.99 |

P(1) = gana el primer equipo · P(X) = empate · P(2) = gana el segundo equipo.

### Tabla esperada del grupo

| Pos | Equipo | Pts esp. | GF-GC esp. | P(1°) | P(2°) | P(3° clasif.) | P(clasificar) |
|:-:|---|:-:|:-:|:-:|:-:|:-:|:-:|
| 1 | **Suiza** | 6.4 | 6.4-2.4 | 53% | 36% | 8% | **97%** |
| 2 | **Canadá** | 6.0 | 5.6-2.5 | 41% | 45% | 10% | **96%** |
| 3 | **Bosnia y Her.** | 3.1 | 3.3-5.0 | 5% | 16% | 35% | **56%** |
| 4 | Catar | 1.3 | 2.2-7.6 | 1% | 4% | 9% | **14%** |

---

## GRUPO C

### Partidos

| Jornada | Fecha | Partido | 🎯 Pronóstico | P(1) | P(X) | P(2) | Goles esperados |
|:-:|---|---|---|:-:|:-:|:-:|:-:|
| J1 | SÁB 13/06 18:00 | **Brasil** vs Marruecos | Gana Brasil 1-0 | 45% | 28% | 27% | 1.44 - 1.07 |
| J1 | SÁB 13/06 21:00 | Haití vs **Escocia** | Gana Escocia 0-1 | 22% | 25% | 52% | 1.03 - 1.71 |
| J2 | VIE 19/06 18:00 | Escocia vs **Marruecos** | Gana Marruecos 0-1 | 18% | 26% | 56% | 0.82 - 1.63 |
| J2 | VIE 19/06 20:30 | **Brasil** vs Haití | Gana Brasil 3-0 | 86% | 10% | 4% | 3.38 - 0.75 |
| J3 | MIÉ 24/06 18:00 | Escocia vs **Brasil** | Gana Brasil 0-2 | 14% | 20% | 66% | 0.97 - 2.29 |
| J3 | MIÉ 24/06 18:00 | **Marruecos** vs Haití | Gana Marruecos 2-0 | 73% | 17% | 10% | 2.46 - 0.79 |

P(1) = gana el primer equipo · P(X) = empate · P(2) = gana el segundo equipo.

### Tabla esperada del grupo

| Pos | Equipo | Pts esp. | GF-GC esp. | P(1°) | P(2°) | P(3° clasif.) | P(clasificar) |
|:-:|---|:-:|:-:|:-:|:-:|:-:|:-:|
| 1 | **Brasil** | 6.5 | 7.1-2.8 | 60% | 29% | 8% | **97%** |
| 2 | **Marruecos** | 5.4 | 5.2-3.0 | 31% | 44% | 15% | **91%** |
| 3 | **Escocia** | 3.2 | 3.5-4.9 | 7% | 20% | 29% | **57%** |
| 4 | Haití | 1.6 | 2.6-7.6 | 1% | 6% | 12% | **19%** |

---

## GRUPO D

### Partidos

| Jornada | Fecha | Partido | 🎯 Pronóstico | P(1) | P(X) | P(2) | Goles esperados |
|:-:|---|---|---|:-:|:-:|:-:|:-:|
| J1 | VIE 12/06 21:00 | **EEUU** vs Paraguay | Gana EEUU 1-0 | 37% | 29% | 34% | 1.27 - 1.21 |
| J1 | DOM 14/06 00:00 | Australia vs **Turquía** | Gana Turquía 0-1 | 34% | 29% | 37% | 1.23 - 1.28 |
| J2 | VIE 19/06 15:00 | **EEUU** vs Australia | Gana EEUU 2-1 | 54% | 24% | 22% | 1.90 - 1.16 |
| J2 | VIE 19/06 23:00 | **Turquía** vs Paraguay | Gana Turquía 1-0 | 45% | 28% | 26% | 1.45 - 1.04 |
| J3 | JUE 25/06 22:00 | Turquía vs **EEUU** | Gana EEUU 1-2 | 35% | 26% | 39% | 1.47 - 1.55 |
| J3 | JUE 25/06 22:00 | **Paraguay** vs Australia | Gana Paraguay 1-0 | 41% | 28% | 30% | 1.41 - 1.17 |

P(1) = gana el primer equipo · P(X) = empate · P(2) = gana el segundo equipo.

### Tabla esperada del grupo

| Pos | Equipo | Pts esp. | GF-GC esp. | P(1°) | P(2°) | P(3° clasif.) | P(clasificar) |
|:-:|---|:-:|:-:|:-:|:-:|:-:|:-:|
| 1 | **EEUU** | 4.7 | 4.7-3.8 | 34% | 27% | 17% | **78%** |
| 2 | **Turquía** | 4.4 | 4.2-3.8 | 28% | 27% | 18% | **74%** |
| 3 | **Paraguay** | 3.9 | 3.7-3.9 | 21% | 25% | 20% | **66%** |
| 4 | **Australia** | 3.4 | 3.6-4.6 | 16% | 21% | 19% | **56%** |

---

## GRUPO E

### Partidos

| Jornada | Fecha | Partido | 🎯 Pronóstico | P(1) | P(X) | P(2) | Goles esperados |
|:-:|---|---|---|:-:|:-:|:-:|:-:|
| J1 | DOM 14/06 13:00 | **Alemania** vs Curazao | Gana Alemania 2-0 | 79% | 14% | 7% | 2.72 - 0.70 |
| J1 | DOM 14/06 19:00 | Costa de Marfil vs **Ecuador** | Gana Ecuador 0-1 | 20% | 30% | 50% | 0.76 - 1.37 |
| J2 | SÁB 20/06 16:00 | **Alemania** vs Costa de Marfil | Gana Alemania 2-0 | 60% | 22% | 17% | 2.05 - 1.01 |
| J2 | SÁB 20/06 20:00 | **Ecuador** vs Curazao | Gana Ecuador 3-0 | 87% | 10% | 4% | 3.29 - 0.62 |
| J3 | JUE 25/06 16:00 | Curazao vs **Costa de Marfil** | Gana Costa de Marfil 0-2 | 6% | 12% | 82% | 0.78 - 3.07 |
| J3 | JUE 25/06 16:00 | Ecuador vs **Alemania** | Gana Alemania 0-1 | 24% | 27% | 49% | 1.04 - 1.58 |

P(1) = gana el primer equipo · P(X) = empate · P(2) = gana el segundo equipo.

### Tabla esperada del grupo

| Pos | Equipo | Pts esp. | GF-GC esp. | P(1°) | P(2°) | P(3° clasif.) | P(clasificar) |
|:-:|---|:-:|:-:|:-:|:-:|:-:|:-:|
| 1 | **Alemania** | 6.3 | 6.4-2.8 | 52% | 31% | 13% | **96%** |
| 2 | **Ecuador** | 5.5 | 5.7-3.0 | 33% | 41% | 21% | **94%** |
| 3 | **Costa de Marfil** | 4.2 | 4.8-4.2 | 14% | 27% | 41% | **82%** |
| 4 | Curazao | 0.8 | 2.1-9.1 | 0% | 2% | 4% | **7%** |

---

## GRUPO F

### Partidos

| Jornada | Fecha | Partido | 🎯 Pronóstico | P(1) | P(X) | P(2) | Goles esperados |
|:-:|---|---|---|:-:|:-:|:-:|:-:|
| J1 | DOM 14/06 16:00 | **Países Bajos** vs Japón | Gana Países Bajos 1-0 | 36% | 31% | 34% | 1.18 - 1.13 |
| J1 | DOM 14/06 22:00 | **Suecia** vs Túnez | Gana Suecia 1-0 | 37% | 28% | 35% | 1.35 - 1.31 |
| J2 | SÁB 20/06 13:00 | **Países Bajos** vs Suecia | Gana Países Bajos 2-0 | 60% | 23% | 17% | 2.01 - 0.99 |
| J2 | DOM 21/06 00:00 | Túnez vs **Japón** | Gana Japón 0-1 | 14% | 24% | 62% | 0.71 - 1.80 |
| J3 | JUE 25/06 19:00 | **Japón** vs Suecia | Gana Japón 1-0 | 51% | 26% | 24% | 1.71 - 1.11 |
| J3 | JUE 25/06 19:00 | Túnez vs **Países Bajos** | Gana Países Bajos 0-2 | 11% | 19% | 69% | 0.82 - 2.27 |

P(1) = gana el primer equipo · P(X) = empate · P(2) = gana el segundo equipo.

### Tabla esperada del grupo

| Pos | Equipo | Pts esp. | GF-GC esp. | P(1°) | P(2°) | P(3° clasif.) | P(clasificar) |
|:-:|---|:-:|:-:|:-:|:-:|:-:|:-:|
| 1 | **Países Bajos** | 5.7 | 5.5-2.9 | 48% | 31% | 12% | **92%** |
| 2 | **Japón** | 5.2 | 4.6-3.0 | 36% | 35% | 15% | **87%** |
| 3 | **Suecia** | 3.1 | 3.4-5.0 | 10% | 19% | 22% | **51%** |
| 4 | Túnez | 2.5 | 2.8-5.4 | 6% | 14% | 19% | **39%** |

---

## GRUPO G

### Partidos

| Jornada | Fecha | Partido | 🎯 Pronóstico | P(1) | P(X) | P(2) | Goles esperados |
|:-:|---|---|---|:-:|:-:|:-:|:-:|
| J1 | LUN 15/06 15:00 | **Bélgica** vs Egipto | Gana Bélgica 1-0 | 56% | 26% | 18% | 1.66 - 0.83 |
| J1 | LUN 15/06 21:00 | **Irán** vs Nueva Zelanda | Gana Irán 1-0 | 55% | 26% | 19% | 1.68 - 0.89 |
| J2 | DOM 21/06 15:00 | **Bélgica** vs Irán | Gana Bélgica 2-0 | 62% | 23% | 15% | 1.94 - 0.86 |
| J2 | DOM 21/06 21:00 | Nueva Zelanda vs **Egipto** | Gana Egipto 0-1 | 20% | 25% | 55% | 0.97 - 1.75 |
| J3 | VIE 26/06 23:00 | **Egipto** vs Irán | Gana Egipto 1-0 | 39% | 30% | 31% | 1.23 - 1.08 |
| J3 | VIE 26/06 23:00 | Nueva Zelanda vs **Bélgica** | Gana Bélgica 0-2 | 10% | 17% | 73% | 0.81 - 2.46 |

P(1) = gana el primer equipo · P(X) = empate · P(2) = gana el segundo equipo.

### Tabla esperada del grupo

| Pos | Equipo | Pts esp. | GF-GC esp. | P(1°) | P(2°) | P(3° clasif.) | P(clasificar) |
|:-:|---|:-:|:-:|:-:|:-:|:-:|:-:|
| 1 | **Bélgica** | 6.4 | 6.1-2.5 | 65% | 22% | 9% | **96%** |
| 2 | **Egipto** | 4.1 | 3.8-3.7 | 17% | 35% | 21% | **73%** |
| 3 | **Irán** | 3.8 | 3.6-4.1 | 14% | 30% | 24% | **68%** |
| 4 | Nueva Zelanda | 2.2 | 2.7-5.9 | 4% | 13% | 14% | **31%** |

---

## GRUPO H

### Partidos

| Jornada | Fecha | Partido | 🎯 Pronóstico | P(1) | P(X) | P(2) | Goles esperados |
|:-:|---|---|---|:-:|:-:|:-:|:-:|
| J1 | LUN 15/06 12:00 | **España** vs Cabo Verde | Gana España 3-0 | 86% | 10% | 4% | 3.00 - 0.50 |
| J1 | LUN 15/06 18:00 | Arabia S. vs **Uruguay** | Gana Uruguay 0-1 | 13% | 25% | 61% | 0.64 - 1.68 |
| J2 | DOM 21/06 12:00 | **España** vs Arabia S. | Gana España 3-0 | 87% | 9% | 3% | 3.24 - 0.58 |
| J2 | DOM 21/06 18:00 | **Uruguay** vs Cabo Verde | Gana Uruguay 1-0 | 67% | 22% | 11% | 1.93 - 0.65 |
| J3 | VIE 26/06 20:00 | Cabo Verde vs **Arabia S.** | Gana Arabia S. 1-2 | 33% | 25% | 42% | 1.48 - 1.68 |
| J3 | VIE 26/06 20:00 | Uruguay vs **España** | Gana España 0-2 | 17% | 24% | 59% | 0.89 - 1.84 |

P(1) = gana el primer equipo · P(X) = empate · P(2) = gana el segundo equipo.

### Tabla esperada del grupo

| Pos | Equipo | Pts esp. | GF-GC esp. | P(1°) | P(2°) | P(3° clasif.) | P(clasificar) |
|:-:|---|:-:|:-:|:-:|:-:|:-:|:-:|
| 1 | **España** | 7.4 | 8.1-2.0 | 79% | 18% | 2% | **99%** |
| 2 | **Uruguay** | 5.1 | 4.5-3.1 | 18% | 58% | 12% | **88%** |
| 3 | Arabia S. | 2.4 | 2.9-6.4 | 2% | 13% | 21% | **36%** |
| 4 | Cabo Verde | 2.0 | 2.6-6.6 | 1% | 10% | 17% | **28%** |

---

## GRUPO I

### Partidos

| Jornada | Fecha | Partido | 🎯 Pronóstico | P(1) | P(X) | P(2) | Goles esperados |
|:-:|---|---|---|:-:|:-:|:-:|:-:|
| J1 | MAR 16/06 15:00 | **Francia** vs Senegal | Gana Francia 1-0 | 55% | 26% | 19% | 1.70 - 0.90 |
| J1 | MAR 16/06 18:00 | Irak vs **Noruega** | Gana Noruega 0-2 | 12% | 22% | 66% | 0.68 - 1.93 |
| J2 | LUN 22/06 17:00 | **Francia** vs Irak | Gana Francia 3-0 | 86% | 10% | 4% | 3.12 - 0.56 |
| J2 | LUN 22/06 20:00 | **Noruega** vs Senegal | Gana Noruega 1-0 | 42% | 27% | 30% | 1.48 - 1.22 |
| J3 | VIE 26/06 15:00 | Noruega vs **Francia** | Gana Francia 0-1 | 24% | 26% | 50% | 1.12 - 1.70 |
| J3 | VIE 26/06 15:00 | **Senegal** vs Irak | Gana Senegal 2-0 | 69% | 20% | 11% | 2.25 - 0.79 |

P(1) = gana el primer equipo · P(X) = empate · P(2) = gana el segundo equipo.

### Tabla esperada del grupo

| Pos | Equipo | Pts esp. | GF-GC esp. | P(1°) | P(2°) | P(3° clasif.) | P(clasificar) |
|:-:|---|:-:|:-:|:-:|:-:|:-:|:-:|
| 1 | **Francia** | 6.4 | 6.5-2.6 | 60% | 27% | 10% | **97%** |
| 2 | **Noruega** | 4.7 | 4.5-3.6 | 22% | 37% | 23% | **83%** |
| 3 | **Senegal** | 4.3 | 4.4-4.0 | 17% | 32% | 30% | **78%** |
| 4 | Irak | 1.3 | 2.0-7.3 | 1% | 5% | 8% | **13%** |

---

## GRUPO J

### Partidos

| Jornada | Fecha | Partido | 🎯 Pronóstico | P(1) | P(X) | P(2) | Goles esperados |
|:-:|---|---|---|:-:|:-:|:-:|:-:|
| J1 | MAR 16/06 21:00 | **Argentina** vs Argelia | Gana Argentina 1-0 | 59% | 24% | 16% | 1.79 - 0.83 |
| J1 | MIÉ 17/06 00:00 | **Austria** vs Jordania | Gana Austria 1-0 | 49% | 26% | 25% | 1.61 - 1.07 |
| J2 | LUN 22/06 13:00 | **Argentina** vs Austria | Gana Argentina 2-0 | 62% | 24% | 14% | 1.89 - 0.79 |
| J2 | LUN 22/06 23:00 | Jordania vs **Argelia** | Gana Argelia 0-2 | 17% | 23% | 60% | 0.95 - 1.97 |
| J3 | SÁB 27/06 22:00 | Argelia vs **Austria** | Gana Austria 0-1 | 34% | 29% | 37% | 1.18 - 1.24 |
| J3 | SÁB 27/06 22:00 | Jordania vs **Argentina** | Gana Argentina 0-2 | 7% | 14% | 78% | 0.78 - 2.79 |

P(1) = gana el primer equipo · P(X) = empate · P(2) = gana el segundo equipo.

### Tabla esperada del grupo

| Pos | Equipo | Pts esp. | GF-GC esp. | P(1°) | P(2°) | P(3° clasif.) | P(clasificar) |
|:-:|---|:-:|:-:|:-:|:-:|:-:|:-:|
| 1 | **Argentina** | 6.6 | 6.5-2.4 | 69% | 21% | 7% | **97%** |
| 2 | **Argelia** | 4.1 | 4.0-4.0 | 15% | 35% | 22% | **73%** |
| 3 | **Austria** | 3.8 | 3.6-4.1 | 13% | 32% | 22% | **67%** |
| 4 | Jordania | 2.1 | 2.8-6.4 | 3% | 12% | 14% | **30%** |

---

## GRUPO K

### Partidos

| Jornada | Fecha | Partido | 🎯 Pronóstico | P(1) | P(X) | P(2) | Goles esperados |
|:-:|---|---|---|:-:|:-:|:-:|:-:|
| J1 | MIÉ 17/06 13:00 | **Portugal** vs RD Congo | Gana Portugal 1-0 | 63% | 24% | 13% | 1.75 - 0.64 |
| J1 | MIÉ 17/06 22:00 | Uzbekistán vs **Colombia** | Gana Colombia 0-2 | 10% | 21% | 69% | 0.65 - 2.07 |
| J2 | MAR 23/06 13:00 | **Portugal** vs Uzbekistán | Gana Portugal 2-0 | 76% | 16% | 8% | 2.51 - 0.71 |
| J2 | MAR 23/06 22:00 | **Colombia** vs RD Congo | Gana Colombia 2-0 | 66% | 22% | 12% | 1.95 - 0.71 |
| J3 | SÁB 27/06 19:30 | Colombia vs **Portugal** | Gana Portugal 0-1 | 32% | 27% | 41% | 1.23 - 1.45 |
| J3 | SÁB 27/06 19:30 | **RD Congo** vs Uzbekistán | Gana RD Congo 2-1 | 40% | 27% | 33% | 1.49 - 1.35 |

P(1) = gana el primer equipo · P(X) = empate · P(2) = gana el segundo equipo.

### Tabla esperada del grupo

| Pos | Equipo | Pts esp. | GF-GC esp. | P(1°) | P(2°) | P(3° clasif.) | P(clasificar) |
|:-:|---|:-:|:-:|:-:|:-:|:-:|:-:|
| 1 | **Portugal** | 6.1 | 5.7-2.6 | 51% | 35% | 9% | **95%** |
| 2 | **Colombia** | 5.7 | 5.2-2.8 | 41% | 41% | 11% | **92%** |
| 3 | RD Congo | 2.7 | 2.8-5.1 | 5% | 14% | 24% | **44%** |
| 4 | Uzbekistán | 2.2 | 2.7-6.1 | 3% | 10% | 18% | **32%** |

---

## GRUPO L

### Partidos

| Jornada | Fecha | Partido | 🎯 Pronóstico | P(1) | P(X) | P(2) | Goles esperados |
|:-:|---|---|---|:-:|:-:|:-:|:-:|
| J1 | MIÉ 17/06 16:00 | **Inglaterra** vs Croacia | Gana Inglaterra 1-0 | 51% | 27% | 21% | 1.55 - 0.92 |
| J1 | MIÉ 17/06 19:00 | Ghana vs **Panamá** | Gana Panamá 0-1 | 21% | 26% | 53% | 1.02 - 1.74 |
| J2 | MAR 23/06 16:00 | **Inglaterra** vs Ghana | Gana Inglaterra 2-0 | 78% | 15% | 7% | 2.62 - 0.66 |
| J2 | MAR 23/06 19:00 | Panamá vs **Croacia** | Gana Croacia 0-1 | 16% | 23% | 61% | 0.84 - 1.89 |
| J3 | SÁB 27/06 17:00 | Panamá vs **Inglaterra** | Gana Inglaterra 0-2 | 10% | 16% | 73% | 0.94 - 2.68 |
| J3 | SÁB 27/06 17:00 | **Croacia** vs Ghana | Gana Croacia 2-0 | 62% | 23% | 15% | 1.99 - 0.88 |

P(1) = gana el primer equipo · P(X) = empate · P(2) = gana el segundo equipo.

### Tabla esperada del grupo

| Pos | Equipo | Pts esp. | GF-GC esp. | P(1°) | P(2°) | P(3° clasif.) | P(clasificar) |
|:-:|---|:-:|:-:|:-:|:-:|:-:|:-:|
| 1 | **Inglaterra** | 6.7 | 6.8-2.5 | 66% | 25% | 6% | **97%** |
| 2 | **Croacia** | 5.1 | 4.8-3.3 | 26% | 46% | 15% | **87%** |
| 3 | **Panamá** | 3.0 | 3.5-5.6 | 6% | 19% | 26% | **52%** |
| 4 | Ghana | 1.9 | 2.6-6.4 | 2% | 10% | 15% | **26%** |
