# Simulation Modules

`simulation/` is the research-code layer for simulator experiments, diagnostics,
and reusable analysis routines.

Keep importable model, estimator, and experiment logic here. Put rendered output
in `figures/`, durable run data in `data/market_impact_study/`, and generated
audit summaries in `artifacts/`.

Common labels:

- `section4_*`: mechanics and immediate impact.
- `section5_*`, `metaorder_*`, `propagation_*`: terminal impact, sliced
  execution, and propagation.
- `section6_*`, `civ_*`, `*_iv_*`: CIV identification and estimator studies.
- `native_*`: native event-engine stress tests.
- `external_lob_*`, `independent_lob_*`: external/public LOB validation.
