# TrackIQ Full App View

Canonical local artifact guide: `artifacts/reports/FULL_APP_FEATURE_VIEW.md`

This repository currently generates and uses these artifact classes:

- `TrackiqResult JSON`: canonical machine contract for automation, comparisons, and gates.
- `HTML reports`: shareable visual summaries for performance and health reviews.
- `CSV/JSON report-data exports`: reproducibility and external analytics integration.
- `UI PNG captures`: UX verification and release/demo collateral.
- Specialized diagnostics:
  - Heatmaps for worker/rank straggler localization.
  - Fault timelines for injection/detection latency validation.
  - Comparison reports for regression and consistency analysis.

## Why Results Are Generated

- Latency percentiles (`p50/p95/p99`) show responsiveness and tail-risk behavior.
- Throughput metrics show capacity and scaling behavior.
- Power/thermal metrics show efficiency and throttling risk.
- All-reduce timing metrics show distributed communication health and straggler risk.
- Variance/consistency findings catch regressions that average metrics can hide.

## Generated Artifact Locations

- Reports: `artifacts/reports/`
- UI screenshots: `artifacts/ui_png/`
- Supporting synthetic inputs: `artifacts/tmp/`

For full per-file purpose mapping and regeneration commands, open:
- `artifacts/reports/FULL_APP_FEATURE_VIEW.md`

