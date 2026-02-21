# MiniCluster Cluster Health Monitor

## What It Does
The MiniCluster health monitor reads incremental `HealthCheckpoint` files while training is running and turns them into live worker-health insights. It flags anomalies like slow workers, failed workers, and communication spikes in near real time. It also produces HTML and JSON health reports for engineering triage and CI pipelines.

## How It Works
`distributed_runner.py` writes checkpoint snapshots incrementally during training. `HealthReader` polls the checkpoint file and deserializes the latest worker state. `AnomalyDetector` classifies anomalies from worker snapshots, and the dashboard/reporting layers render those findings for operators and automated systems.

## Usage
Run a monitored job in one terminal:

```bash
minicluster run --workers 2 --steps 100 --health-checkpoint-path ./minicluster_results/health.json --output ./minicluster_results/run_metrics.json
```

Start monitor in a second terminal:

```bash
minicluster monitor watch --checkpoint ./minicluster_results/health.json
```

Generate HTML report after or during a run:

```bash
minicluster monitor report --checkpoint ./minicluster_results/health.json --output ./minicluster_results/health_report.html
```

Print JSON report for CI:

```bash
minicluster monitor report --checkpoint ./minicluster_results/health.json --json
```

Print one-line status:

```bash
minicluster monitor status --checkpoint ./minicluster_results/health.json
```

## Anomaly Types Reference

| Anomaly Type | Severity | What It Indicates |
|---|---|---|
| `slow_worker` | warning | Worker throughput is materially below cluster peers (straggler risk). |
| `failed_worker` | critical | Worker throughput is zero (process failure or hard stall). |
| `loss_divergence` | warning | Worker loss is statistically high vs peers (potential data/optimizer inconsistency). |
| `allreduce_spike` | warning | Communication latency spike relative to peers (fabric or sync bottleneck). |
| `stalled_worker` | critical | Worker step progress is flat across consecutive checkpoints (deadlock/starvation risk). |

## Mapping to Production
`HealthReader` maps to the telemetry ingestion loop used in cluster observability stacks. `AnomalyDetector` maps directly to the triage phase AMD cluster validation engineers perform when a customer reports flaky failures at scale (for example, intermittent failures at 128 GPUs). `HealthReporter` and the live dashboard map to incident communication artifacts and runbooks that summarize severity, scope, and next investigative actions.

