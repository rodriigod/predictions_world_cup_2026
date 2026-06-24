# B. CLV tracking — closing line value

Registra, por partido, tu probabilidad y las cuotas de **apertura** y **cierre** del mercado para calcular el CLV — el mejor proxy de *edge* real (mejor que la accuracy: te dice si llegaste a la línea antes que el mercado).

## Estado

- Partidos en `results/clv_tracking.csv`: **0**.
- Con CLV calculado (apertura+cierre): **0**.
- Aún sin partidos cerrados: el CLV se valida con el tiempo, partido a partido, sobre amistosos/Mundial reales.

## Fuente de cuotas y LIMITACIONES (honesto)

- **Fuente**: The Odds API (the-odds-api.com), tier gratuito, vía `ODDS_API_KEY` en `.env`. **Hoy NO hay clave configurada**, así que la captura automática de cuotas está INACTIVA: las funciones de red devuelven `None` y el CSV solo se llena con cuotas pasadas a mano. No se inventan cuotas.
- **Cobertura**: el tier gratuito (~500 req/mes) cubre selecciones solo cuando hay torneo/amistoso listado por las casas EU/UK; muchos amistosos menores **no** aparecen.
- **Mercado**: solo **1X2 (h2h)**. No incluye over/under de goles ni hándicaps — el CLV aquí es exclusivamente del resultado.
- **Latencia/'apertura'**: el tier gratuito devuelve la cuota VIGENTE, no un histórico de aperturas. La 'apertura' que se guarda es la cuota al MOMENTO de predecir (early line) y el 'cierre' la del momento previo al partido: aproxima open/close con dos snapshots, no con el tick real de apertura de cada casa.
- **Des-margining**: las cuotas se des-marginan (Shin) y se promedian en log entre casas (`core/data/odds_tools.py`) antes de comparar con tu probabilidad, para no confundir margen de la casa con edge.

## Alternativas si se quiere cobertura real

- Registrar una `ODDS_API_KEY` gratuita y correr `--open`/`--close` en los momentos correctos (idealmente automatizado cerca del kickoff).
- O cargar cuotas a mano con `--odds 1 X 2` desde cualquier casa accesible (Pinnacle es el estándar de oro para CLV por su bajo margen).