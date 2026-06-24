# C. Integración del dataset martj42 — diagnóstico (sin merge)

Fuente: github.com/martj42/international_results (`results.csv`, bajado a `files/f0_raw/martj42_results_latest.csv`). Comparado contra el dataset que core/ ya usa (`files/f0_raw/international_results.csv`).

## Conteos

- Dataset actual de core/: **49,477 partidos** (1872→2026-06-27). core/ entrena desde 1995-01-01 y calienta ELO/forma desde 1980-01-01.
- martj42 (descarga fresca): **49,477 partidos**.
- **Partidos NETOS nuevos** (anti-join fecha+equipos+marcador): **34**.
- Solo en el actual y no en la descarga: **34**.
- Rango temporal de los netos nuevos: 2026-06-12 → 2026-06-20.
- Torneos de los netos nuevos: FIFA World Cup: 34.

## Lectura

El archivo que core/ ya usa **es** martj42 (lo dice su docstring), por lo que el delta es prácticamente nulo: la descarga **confirma** que el set está al día, no aporta partidos relevantes nuevos.

La copia fresca queda en `files/f0_raw/martj42_results_latest.csv` como **fuente candidata** (NO se mezcló al pipeline). Para adoptarla bastaría reemplazar `international_results.csv` y reconstruir la caché del dataset (`files/cache/ensemble/backtest_dataset.csv`).

## Nomenclatura de selecciones (riesgo para IDs)

- Selecciones históricas/desaparecidas presentes: Czechoslovakia, South Yemen, Yemen DPR, Yugoslavia.
- Ninguna de ellas está en `NAME_MAP` (que mapea ES→EN solo para las 48 del Mundial 2026), así que **no rompen los IDs**: core/ las usa solo para calcular ELO/forma histórica (por nombre de cadena, sin ID numérico), y nunca se cruzan con el fixture 2026.
- Las 48 selecciones del Mundial 2026 (vía NAME_MAP) están todas presentes en el dataset por su nombre actual.

> El ELO/forma de core/ se acumula por **nombre actual** de la selección; las entidades sucesoras (p.ej. Germany hereda de West Germany solo si comparten nombre) NO se fusionan automáticamente — es una limitación conocida y honesta de usar el nombre como clave.