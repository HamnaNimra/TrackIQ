# TrackIQ Monorepo TODO

This file is the canonical execution backlog for the `trackiq` repository.

Status legend:

- `[P0]` = critical for reliability/demo readiness
- `[P1]` = high-value next build items
- `[P2]` = medium-term platform upgrades
- `[P3]` = long-term roadmap
- `[OPEN]`, `[IN PROGRESS]`, `[DONE]`, `[BLOCKED]`

---

## Current Focus (Next 1-2 Sprints)

### 1) End-to-end reliability on real outputs `[P0] [OPEN]`

Goal: every tool generates real `TrackiqResult` files that render cleanly in dashboards and compare flows.

Tasks:

1. Run `autoperfpy` real run -> save JSON in `output/`.
2. Run `minicluster` real run -> save JSON in `minicluster_results/`.
3. Run `trackiq-compare run` on both files (terminal + HTML).
4. Load all three dashboards with real files and fix any rendering issues.
5. Record a known-good demo command list in root `README.md`.

Acceptance criteria:

- No crashes in run/compare/dashboard path with real files.
- No placeholder-only sections when real data exists.
- Comparison output includes correct winner logic for power metrics.

---

## High Priority Backlog

### 2) LLM metrics wired into canonical schema `[P1] [Done]`

Problem: LLM/KV concepts exist, but schema-first integration is incomplete.

Tasks:

1. Add explicit LLM fields to schema (for example `ttft_ms`, `tokens_per_sec`, KV cache metrics).
2. Wire `LLMKVCacheMonitor` output into `TrackiqResult.tool_payload` and/or schema metrics.
3. Update serializer/validator tests for new fields.
4. Update dashboards to display LLM metrics when present.

Acceptance criteria:

- LLM run produces non-null LLM fields.
- `trackiq-compare` can compare LLM-specific metrics without code changes beyond winner rules.

### 3) Multi-run trend analysis `[P1] [Done]`

Problem: current comparison is mostly pairwise; historical trends across many runs are missing.

Tasks:

1. Add run-history loader (directory of `TrackiqResult` JSON files).
2. Build trend component in `trackiq_core/ui` for metric-over-time.
3. Add trend section to dashboards (at least autoperfpy + compare).
4. Add tests with synthetic run history.

Acceptance criteria:

- Plot at least throughput, p99 latency, and perf/watt over time.
- Supports at least 10+ runs without UI failure.

### 4) PDF report generation consistency `[P1] [DONE]`

Problem: HTML is strong; PDF path needs standardization and reliability.

Tasks:

1. Standardize one PDF backend and fallback behavior.
2. Make report commands deterministic across tools.
3. Add smoke tests for PDF generation in CI-appropriate mode.

Acceptance criteria:

- Each tool can generate PDF from canonical result input.
- Clear error message when system dependency is missing.

### 4.1) Consolidated graphs and visualizations `[P1] [OPEN]`

Problem: current charts are strong per-metric views, but we still need consolidated visuals that summarize run health quickly and compare runs with less manual tab-switching.

Tasks:

1. Add a shared "Overview" chart pack in `autoperfpy/reports/charts.py` for both Streamlit and HTML report paths.
2. Add an "overview timeline" chart combining latency, throughput, and power/utilization (normalized index or dual-axis where appropriate).
3. Add a "run health scorecard" visualization (table + bars) computed from p99 latency, mean throughput, perf/watt, and variability.
4. Add a consolidated multi-run heatmap (`run x metric`) for p50/p95/p99 latency, mean throughput, mean power, mean CPU/GPU, and mean memory.
5. Add correlation/scatter views:
   - latency vs throughput
   - throughput vs power (efficiency frontier)
   - GPU utilization vs latency (where GPU data exists)
6. Add missing-data coverage view (which metrics were present per run) to avoid misleading comparisons.
7. Wire the same consolidated pack into:
   - CLI HTML report generation
   - Streamlit report export
   - Streamlit in-app comparison page
8. Add tests for consolidated builders (non-empty figures, expected section names, and graceful behavior when metrics are missing).

Candidate visualizations to add:

1. Overview KPI strip (already partly available; consolidate and standardize labels/units).
2. Normalized trend chart (`0-100`) for latency/throughput/power over elapsed time.
3. Multi-run metric heatmap (`runs x metrics`).
4. Efficiency frontier scatter (`power_w` vs `throughput_fps`, colored by run/device).
5. Regression delta bars against baseline (`% change` for key metrics).
6. Data completeness matrix (metric presence by run).

Implementation plan:

1. Sprint 1 (MVP):
   - Add shared overview pack + multi-run heatmap.
   - Add baseline delta bars for runs with baseline selected.
   - Wire into HTML report + Streamlit report export.
2. Sprint 2:
   - Add correlation/scatter pack and data completeness matrix.
   - Add filtering controls in Streamlit (metric subset, run subset, normalize on/off).
3. Sprint 3:
   - Add ranking/scorecard view and optional weighted scoring config.
   - Add PDF parity checks for consolidated sections.

Acceptance criteria:

- Same input data produces equivalent consolidated sections in CLI-generated HTML and Streamlit-generated HTML export.
- Consolidated charts render with partial metrics (no crashes; unavailable metrics are omitted with clear note).
- Multi-run comparison supports at least 10 runs and still renders within acceptable UI time.
- Tests cover at least one full-metric fixture and one sparse-metric fixture.

---

## Platform Expansion

### 5) Precision expansion `[P2] [DONE]`

Add BF16, INT4, and mixed precision modes where supported.

Tasks:

1. Extend precision constants/config.
2. Add capability checks per device.
3. Update CLI validation and profile compatibility checks.
4. Add tests for unsupported precision fallback.

### 6) Accelerator categories and capability-aware runs `[P2] [OPEN]`

Support explicit accelerator classes and constraints.

Targets:

- GPU
- NPU/DLA
- DSP
- FPGA

Tasks:

1. Extend device detection metadata/categorization.
2. Add accelerator-aware profile constraints.
3. Add CLI filtering by accelerator type.
4. Ensure runner skips unsupported config combinations safely.

### 7) Qualcomm support path `[P2] [OPEN]`

Goal: add concrete detection + telemetry strategy for Qualcomm platforms.

Tasks:

1. Add Qualcomm device profile and detection hooks.
2. Define first metrics source (tooling/API availability dependent).
3. Add compare/dashboard support through existing schema path.

---

## Monitoring and API

### 8) Health monitor hardening `[P1] [OPEN]`

Current polling flow is good; now harden for production usage.

Tasks:

1. Add checkpoint schema versioning.
2. Add stale checkpoint and timeout diagnostics.
3. Add monitor CLI UX polish for CI consumption.

### 9) REST API for live monitoring `[P3] [OPEN]`

Long-term: API-based monitoring in addition to file checkpoints.

Tasks:

1. Define API contract for run status, worker state, anomalies.
2. Build minimal service layer (read-only first).
3. Add auth and deployment guidance later.

---

## Documentation and Product Story

### 10) Case study + docs alignment `[P0] [IN PROGRESS]`

Goal: docs reflect current monorepo reality (`trackiq_core` + 3 tools).

Tasks:

1. Keep root README as monorepo source of truth.
2. Keep case study architecture/roadmap aligned with shipped features.
3. Maintain a short "what is shipped vs next" section in docs.

Acceptance criteria:

- No stale architecture claims.
- No broken links/path instructions.
- No encoding artifacts in docs.

### 11) Domain examples pack `[P2] [OPEN]`

Add worked examples for:

1. Automotive edge inference
2. ROCm/driver comparison
3. Cluster training health monitoring

---

## Quality and Testing

### 12) Integration test expansion `[P1] [OPEN]`

Tasks:

1. End-to-end test from run -> save -> compare -> dashboard load checks (non-Streamlit logic).
2. Add golden-file comparison tests for serializer stability.
3. Add performance/power metric sanity checks.

### 13) CI matrix and environment checks `[P2] [OPEN]`

Tasks:

1. Validate Python 3.12 as primary baseline.
2. Add optional hardware-aware test groups (skip when unavailable).
3. Add docs lints/encoding checks to prevent mojibake regressions.

---

## Refactoring and Code Health

### 14) CLI decomposition and boundary cleanup `[P1] [OPEN]`

Problem: very large command modules increase regression risk and reduce maintainability.

Tasks:

1. Split `autoperfpy/cli.py` into command modules (`run`, `report`, `analyze`, `monitor`).
2. Keep parser wiring thin and move command logic into testable functions.
3. Add shared save/serialization helpers to remove duplicate result-writing paths.
4. Add import-cycle checks between tool packages and `trackiq_core`.

Acceptance criteria:

- No CLI behavior regressions.
- Reduced complexity and better unit-test coverage per command area.

### 15) Schema-first contract enforcement `[P1] [OPEN]`

Tasks:

1. Validate all final tool outputs with `validate_trackiq_result_obj`.
2. Add contract tests for autoperfpy/minicluster/compare outputs.
3. Add backward-compatibility tests for loading older schema versions.

Acceptance criteria:

- Schema contract breaks are caught in CI before merge.

### 16) Technical debt register `[P2] [OPEN]`

Tasks:

1. Track temporary shims/workarounds with owner + removal milestone.
2. Add a quarterly debt burn-down checkpoint.
3. Enforce "no new TODO without owner/date" rule in PR reviews.

### 16.1) Refactor and dead-code cleanup sweep `[P1] [OPEN]`

Problem: incremental feature delivery has left stale helpers/imports and partially duplicated logic that increase maintenance and regression risk.

Tasks:

1. Run static checks for unused imports, variables, functions, and unreachable paths.
2. Remove dead modules/helpers and eliminate duplicate utility paths across `trackiq_core` and tool packages.
3. Refactor oversized modules where cleanup exposes boundary issues.
4. Add or update tests in every touched area to preserve behavior during cleanup.
5. Add a CI guard for unused code/import regressions.

Acceptance criteria:

- No behavior regressions in run/save/compare/dashboard flows.
- Measurable reduction in unused-code lint findings.
- Changelog note describing removed deprecated/stale paths.

### 16.2) Optional debug harness replacements `[P2] [OPEN]`

Problem: dead standalone debug scripts were removed; a maintained replacement may still be useful.

Tasks:

1. Add a supported chart-debug harness under `scripts/` (or `tests/fixtures`) for synthetic sample generation and chart smoke checks.
2. Add a maintained report-debug command that exercises `autoperfpy report html/pdf` using canonical fixture inputs.
3. Add an optional `distributed-validator` smoke CLI wrapper (or documented pytest command) for quick local checks.
4. Decide whether `autoperfpy/cli/utils.py` compatibility wrappers should be retained or removed in a future breaking-change window.

Acceptance criteria:

- Debug workflows are reachable via documented commands (not ad-hoc root scripts).
- No orphan/one-off debug scripts at repository root.

---

## Repository Quality Standards

### 17) Engineering standards baseline `[P1] [OPEN]`

Tasks:

1. Standardize `ruff`, `black`, `isort`, and `mypy` configuration.
2. Add pre-commit hooks for lint/format/import checks.
3. Define minimum docstring/type-hint requirements for new modules.

Acceptance criteria:

- CI blocks non-conforming style/type/lint changes.

### 18) CI quality gates `[P1] [OPEN]`

Tasks:

1. Require tests + coverage threshold + integration smoke tests.
2. Add packaging smoke test (`pip install .` and import core/tool modules).
3. Add CLI smoke test for `autoperfpy`, `minicluster`, `trackiq-compare`.

Acceptance criteria:

- No release-branch merge without passing all gates.

### 19) Release and versioning discipline `[P2] [OPEN]`

Tasks:

1. Adopt semantic versioning rules for repo/tool/schema changes.
2. Add release checklist (compatibility, migrations, docs).
3. Auto-generate changelog entries from PR labels.

### 20) Security and dependency hygiene `[P2] [OPEN]`

Tasks:

1. Add dependency audit checks in CI.
2. Add monthly dependency update cadence.
3. Ensure dependency parity between `pyproject.toml` and `requirements.txt`.

### 21) Observability and diagnostics `[P2] [OPEN]`

Tasks:

1. Standardize structured logging across tools.
2. Add run/correlation IDs to reports and monitor outputs.
3. Improve actionable error messages for dependency/hardware failures.

---

## Immediate Action Plan (This Week)

1. `[P0]` Run real `autoperfpy`, `minicluster`, and `trackiq-compare` flows and capture artifacts.
2. `[P0]` Fix any dashboard rendering gaps exposed by those artifacts.
3. `[P1]` Start LLM schema integration (`ttft_ms`, KV metrics) behind additive fields.
4. `[P1]` Implement initial multi-run trend component in `trackiq_core/ui`.
5. `[P0]` Keep docs/case study synchronized with shipped behavior after each merge.
6. `[P1]` Deliver consolidated visualization MVP (overview pack + multi-run heatmap + baseline deltas) across CLI HTML and Streamlit export.
