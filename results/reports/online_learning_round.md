# E. Online learning — reporte por ronda

Partidos 2026 cargados: **36** (evaluados: 36).

## Original vs Actualizado (sobre los partidos jugados)

| Modelo | Accuracy | RPS |
|---|:-:|:-:|
| ensemble ORIGINAL (pre-torneo) | 0.556 | 0.1775 |
| ensemble ACTUALIZADO (2026) | 0.639 | 0.1578 |

> Con 36 partido(s), estas cifras son anecdóticas (no concluyentes): el valor del online learning se mide a lo largo del torneo, no con la primera ronda.

## ELO: top movimientos tras la ronda

| ↑ Subieron | Δ ELO | | ↓ Bajaron | Δ ELO |
|---|:-:|---|---|:-:|
| EEUU (1823→1861) | +37.6 | | Ecuador (2018→1983) | -35.3 |
| Ghana (1629→1652) | +23.3 | | Turquía (1951→1918) | -33.3 |
| México (1966→1983) | +16.0 | | Panamá (1846→1823) | -23.3 |
| Costa de Marfil (1807→1822) | +14.7 | | Túnez (1748→1728) | -19.9 |
| Cabo Verde (1649→1663) | +13.7 | | Rep. Checa (1792→1775) | -16.5 |

## Partidos: original vs actualizado

| Partido | Real | ORIG (1/X/2) | ACTUALIZADO (1/X/2) |
|---|:-:|:-:|:-:|
| México – Sudáfrica | 2-0 | 0.78/0.12/0.10 | 0.78/0.12/0.10 |
| Corea del Sur – Rep. Checa | 2-1 | 0.45/0.27/0.28 | 0.48/0.26/0.26 |
| Canadá – Bosnia y Her. | 1-1 | 0.71/0.16/0.12 | 0.75/0.14/0.11 |
| Suiza – Catar | 1-1 | 0.83/0.09/0.08 | 0.83/0.09/0.08 |
| Brasil – Marruecos | 1-1 | 0.44/0.28/0.27 | 0.44/0.29/0.27 |
| Escocia – Haití | 1-0 | 0.53/0.25/0.22 | 0.54/0.24/0.21 |
| EEUU – Paraguay | 4-1 | 0.35/0.31/0.34 | 0.44/0.30/0.27 |
| Australia – Turquía | 2-0 | 0.34/0.30/0.36 | 0.38/0.30/0.32 |
| Alemania – Curazao | 7-1 | 0.80/0.11/0.09 | 0.81/0.10/0.09 |
| Costa de Marfil – Ecuador | 1-0 | 0.22/0.29/0.49 | 0.25/0.32/0.43 |
| Países Bajos – Japón | 2-2 | 0.36/0.31/0.34 | 0.37/0.29/0.34 |
| Suecia – Túnez | 5-1 | 0.37/0.29/0.35 | 0.46/0.26/0.28 |
| Bélgica – Egipto | 1-1 | 0.57/0.23/0.19 | 0.55/0.25/0.21 |
| Irán – Nueva Zelanda | 2-2 | 0.57/0.23/0.20 | 0.55/0.24/0.21 |
| España – Cabo Verde | 0-0 | 0.84/0.08/0.08 | 0.81/0.10/0.09 |
| Arabia S. – Uruguay | 1-1 | 0.17/0.23/0.60 | 0.18/0.24/0.58 |
| Francia – Senegal | 3-1 | 0.58/0.22/0.20 | 0.62/0.20/0.17 |
| Noruega – Irak | 4-1 | 0.68/0.18/0.14 | 0.72/0.15/0.12 |
| Argentina – Argelia | 3-0 | 0.63/0.20/0.17 | 0.68/0.17/0.15 |
| Austria – Jordania | 3-1 | 0.49/0.26/0.25 | 0.54/0.24/0.22 |
| Portugal – RD Congo | 1-1 | 0.66/0.19/0.15 | 0.64/0.20/0.16 |
| Colombia – Uzbekistán | 3-1 | 0.64/0.20/0.16 | 0.68/0.18/0.14 |
| Inglaterra – Croacia | 4-2 | 0.51/0.26/0.23 | 0.57/0.23/0.20 |
| Ghana – Panamá | 1-0 | 0.23/0.25/0.52 | 0.26/0.28/0.46 |
| Rep. Checa – Sudáfrica | 1-1 | 0.55/0.24/0.21 | 0.52/0.25/0.23 |
| Suiza – Bosnia y Her. | 4-1 | 0.74/0.14/0.12 | 0.74/0.14/0.12 |
| Canadá – Catar | 6-0 | 0.81/0.10/0.08 | 0.84/0.09/0.08 |
| México – Corea del Sur | 1-0 | 0.56/0.25/0.19 | 0.57/0.25/0.19 |
| EEUU – Australia | 2-0 | 0.34/0.31/0.35 | 0.42/0.30/0.28 |
| Escocia – Marruecos | 0-1 | 0.22/0.26/0.51 | 0.22/0.27/0.51 |
| Brasil – Haití | 3-0 | 0.78/0.12/0.10 | 0.79/0.11/0.10 |
| Turquía – Paraguay | 0-1 | 0.38/0.29/0.33 | 0.34/0.30/0.36 |
| Países Bajos – Suecia | 5-1 | 0.68/0.17/0.15 | 0.72/0.15/0.13 |
| Alemania – Costa de Marfil | 2-1 | 0.55/0.24/0.21 | 0.55/0.23/0.21 |
| Ecuador – Curazao | 0-0 | 0.77/0.12/0.10 | 0.75/0.14/0.11 |
| Túnez – Japón | 0-4 | 0.18/0.24/0.58 | 0.15/0.17/0.68 |

> Reporte autogenerado por `online_learning`. El módulo es paralelo: NO modifica core/microsim/ensemble.