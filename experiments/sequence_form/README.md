# experiments/sequence_form — forma como secuencia aprendida (D)

Experimento **aislado** (no toca `core/` ni Dixon-Coles): ¿un GRU que codifica los
últimos N partidos de cada selección representa la "forma" mejor que los
pi-ratings que `core/` ya usa?

## Archivos

- `build_dataset.py` — recorre `international_results.csv` una vez y arma, por
  partido usable (era de core/, ambas selecciones con ≥N partidos previos):
  secuencias `(N, 8)` por equipo (resultado, GF/GA, localía, Elo rival/propio),
  los pi-ratings + ΔElo, y la etiqueta 1X2. Anti-leakage (solo pasado).
- `run_experiment.py` — entrena/compara 3 representaciones con TimeSeriesSplit y
  RPS+accuracy: **pi-ratings solo** vs **secuencia (GRU) sola** vs **ambas**.

## Cómo correr

```bash
pip install torch                       # requerido (CPU está bien)
python experiments/sequence_form/run_experiment.py --n-splits 4 --epochs 12
```

Salida: `results/reports/sequence_form_experiment.md`.

## Resultado (resumen)

La secuencia aprendida **no le gana** a los pi-ratings: en N=10 y N=20 las
diferencias de RPS están dentro del ruido (< 0.0015). Un rating online recursivo
ya captura casi toda la señal de forma en selecciones. **No se adopta** en core/;
era una comparación, no un reemplazo. El reporte lo dice tal cual.

> Requiere `torch` (instalado en este entorno: 2.12.1 sobre Python 3.14).
