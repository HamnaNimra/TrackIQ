# TrackIQ UI Layer

## Overview
`trackiq_core.ui` is a shared, library-first dashboard layer for canonical `TrackiqResult` data. It is designed so any team can install `trackiq-core`, subclass one base dashboard class, and ship a styled validation UI with minimal glue code.

## Quick Start
```python
from trackiq_core.serializer import load_trackiq_result
from trackiq_core.ui import TrackiqDashboard, MetricTable, RegressionBadge


class MyDashboard(TrackiqDashboard):
    def render_body(self) -> None:
        result = self.result[0] if isinstance(self.result, list) else self.result
        RegressionBadge(result.regression, theme=self.theme).render()
        MetricTable(result=result, mode="single", theme=self.theme).render()


if __name__ == "__main__":
    result = load_trackiq_result("result.json")
    MyDashboard(result=result).run()
```

## Component Reference
`MetricTable` displays canonical metrics in either single-result mode or two-result comparison mode with deltas and winner indicators. It requires one `TrackiqResult` (single) or exactly two `TrackiqResult` objects (comparison).

`LossChart` plots loss over steps and can optionally overlay a baseline loss series with a tolerance band. It requires `steps` and `loss_values`, and optionally `baseline_values`.

`RegressionBadge` renders a large pass/fail status block plus delta percent using regression metadata. It requires a `RegressionInfo` object.

`WorkerGrid` renders distributed worker cards with throughput, all-reduce time, and health status highlighting. It requires a list of worker dictionaries.

`PowerGauge` renders power and efficiency metrics (mean/peak power, perf per watt, energy per step) and shows a fallback message when power data is unavailable. It requires a `Metrics` object and optionally tool payload data for peak values.

`ComparisonTable` renders end-to-end pairwise comparison including platform differences, metric deltas, and a plain-English winner summary. It requires two `TrackiqResult` objects and optional labels.

## Theme Customization
Create a custom `TrackiqTheme` and pass it into any dashboard or component:

```python
from trackiq_core.ui import TrackiqTheme

MY_THEME = TrackiqTheme(
    name="custom",
    background_color="#101418",
    surface_color="#1B232D",
    text_color="#EAEFF4",
    accent_color="#D32F2F",
    pass_color="#2E7D32",
    fail_color="#C62828",
    warning_color="#F9A825",
    chart_colors=["#D32F2F", "#546E7A", "#78909C", "#90A4AE", "#B0BEC5"],
)
```

## External Usage
After installing:

```bash
pip install trackiq-core
```

you can launch a dashboard from any external repo:

```python
from trackiq_core.ui import TrackiqDashboard, run_dashboard, MetricTable, PowerGauge


class ExternalValidationDashboard(TrackiqDashboard):
    def render_body(self) -> None:
        result = self.result
        MetricTable(result=result, mode="single", theme=self.theme).render()
        PowerGauge(metrics=result.metrics, tool_payload=result.tool_payload, theme=self.theme).render()


run_dashboard(
    dashboard_class=ExternalValidationDashboard,
    result_path="trackiq_result.json",
)
```

