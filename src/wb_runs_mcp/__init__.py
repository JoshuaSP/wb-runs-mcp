"""
wandb-mcp — A minimal, composable W&B MCP server.

Tools:
  list_projects  — discover projects (GET /projects)
  list_runs      — search/filter runs (GET /runs)
  get_run        — full detail for one run (GET /runs/:id)
  get_metrics    — time-series data with auto-downsampling (GET /runs/:id/metrics)
  compare_runs   — side-by-side comparison of 2+ runs (GET /runs/compare)
"""

import json
import math
from typing import Any

import wandb
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("wandb")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _api() -> wandb.Api:
    return wandb.Api()


def _resolve_entity(api: wandb.Api, entity: str | None) -> str:
    if entity:
        return entity
    return api.viewer.entity


def _clean_val(v: Any) -> Any:
    """Clean a value for JSON output."""
    if isinstance(v, float):
        return round(v, 6) if not math.isnan(v) else None
    if isinstance(v, int):
        return v
    if isinstance(v, str):
        if v.lower() == "nan":
            return None
        try:
            return round(float(v), 6)
        except ValueError:
            return v
    return v


def _error(msg: str) -> str:
    return json.dumps({"error": msg})


def _get_history_cols(run, stream: str = "default") -> list[str]:
    """Get all column names from a run's history."""
    h = run.history(samples=2, stream=stream)
    if hasattr(h, "columns"):
        return sorted(h.columns.tolist())
    elif isinstance(h, list) and h:
        return sorted(h[0].keys())
    return sorted(run.summary.keys())


def _metric_stats(values: list[float | int]) -> dict[str, Any]:
    """Compute summary stats for a list of numeric values."""
    if not values:
        return {"min": None, "max": None, "mean": None, "final": None, "count": 0}
    return {
        "min": round(min(values), 6),
        "max": round(max(values), 6),
        "mean": round(sum(values) / len(values), 6),
        "final": round(values[-1], 6),
        "count": len(values),
    }


def _is_system_metric(name: str) -> bool:
    return name.startswith("system.") or name.startswith("system/")


def _fetch_metric_data(run, metrics: list[str], min_step=None, max_step=None, max_points=200) -> list[dict]:
    """Fetch and downsample metric data from a run."""
    # Split metrics by stream — system.* metrics live on a separate stream
    user_metrics = [m for m in metrics if not _is_system_metric(m)]
    sys_metrics = [m for m in metrics if _is_system_metric(m)]

    rows = []
    if user_metrics:
        # Include _timestamp for aligning with system metrics
        fetch_keys = user_metrics if not sys_metrics else user_metrics + ["_timestamp"]
        rows = _fetch_from_stream(run, fetch_keys, "default", min_step, max_step, max_points)

    if sys_metrics:
        sys_rows = _fetch_from_stream(run, sys_metrics, "system", min_step, max_step, max_points)
        if rows:
            # Merge system data into user rows by nearest _runtime
            # User rows have _timestamp, system rows have _runtime — use _timestamp for alignment
            sys_by_ts = {r.get("_timestamp", 0): r for r in sys_rows if r.get("_timestamp")}
            sys_timestamps = sorted(sys_by_ts.keys())
            for row in rows:
                ts = row.get("_timestamp", 0)
                if sys_timestamps and ts:
                    nearest = min(sys_timestamps, key=lambda s: abs(s - ts))
                    for m in sys_metrics:
                        row[m] = sys_by_ts[nearest].get(m)
        else:
            rows = sys_rows

    data_points = []
    for r in rows:
        point: dict[str, Any] = {"step": r.get("_step")}
        for m in metrics:
            point[m] = _clean_val(r.get(m))
        data_points.append(point)
    return data_points


def _fetch_from_stream(run, metrics: list[str], stream: str, min_step, max_step, max_points) -> list[dict]:
    """Fetch metric data from a specific wandb history stream."""
    if stream == "system":
        # System stream uses _runtime not _step, and keys= filter is broken
        # in the wandb SDK — must fetch all columns and filter client-side
        h = run.history(samples=max_points, stream="system")
        if hasattr(h, "to_dict"):
            rows = h.to_dict("records")
        elif isinstance(h, list):
            rows = h
        else:
            rows = []
        # Add a synthetic _step from _runtime for alignment
        for r in rows:
            if "_step" not in r:
                r["_step"] = r.get("_runtime", 0)
        return rows

    if min_step is not None or max_step is not None:
        rows = []
        for r in run.scan_history(keys=metrics + ["_step"]):
            step = r.get("_step", 0)
            if min_step is not None and step < min_step:
                continue
            if max_step is not None and step > max_step:
                break
            rows.append(dict(r))
        if len(rows) > max_points:
            stride = len(rows) / max_points
            rows = [rows[int(i * stride)] for i in range(max_points)]
    else:
        h = run.history(keys=metrics, samples=max_points, stream=stream)
        if hasattr(h, "to_dict"):
            rows = h.to_dict("records")
        elif isinstance(h, list):
            rows = h
        else:
            rows = []
    return rows


# ---------------------------------------------------------------------------
# Tool 1: list_projects
# ---------------------------------------------------------------------------

LIST_PROJECTS_DESC = """\
List W&B projects for an entity (user or team). Use this to discover available \
projects before calling other tools.

Args:
  - entity: W&B username or team name (optional, defaults to your account). \
If omitted, lists projects for your account and all teams you belong to.

Returns project name, entity, description, run count, and last updated time.
"""


@mcp.tool(description=LIST_PROJECTS_DESC)
def list_projects(
    entity: str | None = None,
) -> str:
    try:
        api = _api()
        viewer = api.viewer

        if entity:
            entities = [entity]
        else:
            entities = [viewer.entity] + (viewer.teams or [])

        all_projects = []
        for ent in entities:
            try:
                for p in api.projects(entity=ent):
                    all_projects.append({
                        "name": p.name,
                        "entity": ent,
                        "description": getattr(p, "description", None) or "",
                        "url": f"https://wandb.ai/{ent}/{p.name}",
                    })
            except Exception as e:
                all_projects.append({"entity": ent, "error": str(e)})

        return json.dumps({
            "default_entity": viewer.entity,
            "count": len(all_projects),
            "projects": all_projects,
        }, default=str)
    except Exception as e:
        return _error(f"Failed to list projects: {e}")


# ---------------------------------------------------------------------------
# Tool 2: list_runs
# ---------------------------------------------------------------------------

LIST_RUNS_DESC = """\
Search for W&B runs in a project. Returns a compact list per run.

Sorted newest-first by default.

Filtering (all optional):
  - name_contains: substring match on run display name (case-insensitive)
  - name_regex: regex match on run display name (e.g. "policy.*", "run-0[1-3]")
  - state: exact match — "finished", "running", "crashed", "failed"
  - created_after / created_before: ISO date or datetime (e.g. "2026-03-28")
  - config_filters: dict of config constraints, e.g. {"lr": 0.001}
  - tags: list of required tags (all must match)

Pagination:
  - limit: max runs to return (default 10, max 50)
  - offset: skip this many runs before returning results

Options:
  - include_config: if true, include the full config dict per run (default false)

Common patterns:
  - Latest run: just call with limit=1
  - Currently running: state="running"
  - Find by name: name_contains="policy" or name_regex="policy-val-0[1-3]"
"""


@mcp.tool(description=LIST_RUNS_DESC)
def list_runs(
    project: str,
    entity: str | None = None,
    name_contains: str | None = None,
    name_regex: str | None = None,
    state: str | None = None,
    created_after: str | None = None,
    created_before: str | None = None,
    config_filters: dict[str, Any] | None = None,
    tags: list[str] | None = None,
    include_config: bool = False,
    limit: int = 10,
    offset: int = 0,
) -> str:
    try:
        api = _api()
        entity = _resolve_entity(api, entity)
        limit = min(limit, 50)

        filters: dict[str, Any] = {}
        if name_contains:
            filters["display_name"] = {"$regex": f"(?i){name_contains}"}
        elif name_regex:
            filters["display_name"] = {"$regex": name_regex}
        if state:
            filters["state"] = state
        if created_after or created_before:
            ts: dict[str, str] = {}
            if created_after:
                ts["$gte"] = created_after
            if created_before:
                ts["$lte"] = created_before
            filters["created_at"] = ts
        if config_filters:
            for k, v in config_filters.items():
                filters[f"config.{k}"] = v
        if tags:
            filters["tags"] = {"$all": tags}

        runs = api.runs(
            f"{entity}/{project}",
            filters=filters if filters else None,
            order="-created_at",
            per_page=min(offset + limit, 50),
        )

        results = []
        skipped = 0
        for run in runs:
            if skipped < offset:
                skipped += 1
                continue
            summary = {k: _clean_val(v) for k, v in run.summary.items() if not k.startswith("_")}
            entry: dict[str, Any] = {
                "id": run.id,
                "name": run.name,
                "state": run.state,
                "created_at": run.created_at,
                "tags": run.tags,
                "summary_metrics": summary,
                "url": run.url,
            }
            if include_config:
                entry["config"] = run.config
            results.append(entry)
            if len(results) >= limit:
                break

        return json.dumps({
            "entity": entity,
            "project": project,
            "offset": offset,
            "count": len(results),
            "runs": results,
        }, default=str)
    except Exception as e:
        return _error(f"Failed to list runs: {e}")


# ---------------------------------------------------------------------------
# Tool 3: get_run
# ---------------------------------------------------------------------------

GET_RUN_DESC = """\
Get full detail for a single run: config, all available metric names (grouped by prefix), \
summary/final values, step count, and metadata.

Use this after list_runs to inspect a specific run, or before get_metrics to discover \
what metrics are available.

Args:
  - project: W&B project name
  - run_id: the 8-character run ID (e.g. "m94p3szz")
  - entity: W&B entity/team (optional, defaults to your account)
  - include_system: if true, also list system metrics (GPU util, memory, temp, etc.) \
These are logged on a separate stream by wandb. Default false to keep output compact.

Returns: id, name, state, created_at, tags, url, config, step_count, \
available metrics (grouped by prefix), and summary values (final value of each metric).
"""


@mcp.tool(description=GET_RUN_DESC)
def get_run(
    project: str,
    run_id: str,
    entity: str | None = None,
    include_system: bool = False,
) -> str:
    try:
        api = _api()
        entity = _resolve_entity(api, entity)
        run = api.run(f"{entity}/{project}/{run_id}")

        all_cols = _get_history_cols(run)
        user_metrics = [c for c in all_cols if not c.startswith("_")]

        grouped: dict[str, list[str]] = {}
        for m in user_metrics:
            prefix = m.split("/")[0] if "/" in m else "(ungrouped)"
            grouped.setdefault(prefix, []).append(m)

        # System metrics (GPU util, memory, temp, etc.) are on a separate stream
        if include_system:
            sys_cols = _get_history_cols(run, stream="system")
            sys_metrics = [c for c in sys_cols if not c.startswith("_")]
            sys_grouped: dict[str, list[str]] = {}
            for m in sys_metrics:
                prefix = m.split(".")[0] if "." in m else "(ungrouped)"
                sys_grouped.setdefault(prefix, []).append(m)
            for prefix, metrics in sys_grouped.items():
                grouped[f"system/{prefix}"] = metrics

        summary = {k: _clean_val(v) for k, v in run.summary.items() if not k.startswith("_")}

        return json.dumps({
            "id": run.id,
            "name": run.name,
            "state": run.state,
            "created_at": run.created_at,
            "tags": run.tags,
            "url": run.url,
            "config": run.config,
            "step_count": run.summary.get("_step"),
            "metrics": grouped,
            "total_metrics": len(user_metrics),
            "summary": summary,
        }, default=str)
    except Exception as e:
        return _error(f"Failed to get run '{run_id}': {e}")


# ---------------------------------------------------------------------------
# Tool 4: get_metrics
# ---------------------------------------------------------------------------

GET_METRICS_DESC = """\
Get time-series data for specific metrics on a run.

Args:
  - project: W&B project name
  - run_id: the 8-character run ID
  - metrics: list of metric names (e.g. ["train/loss", "eval/mae_999"])
  - entity: W&B entity/team (optional, defaults to your account)
  - min_step / max_step: optional step range to zoom into a region
  - max_points: cap on returned data points (default 200, hard max 500). Auto-downsamples.

Returns data points (step + values) plus summary stats (min, max, mean, final) per metric.
"""


@mcp.tool(description=GET_METRICS_DESC)
def get_metrics(
    project: str,
    run_id: str,
    metrics: list[str],
    entity: str | None = None,
    min_step: int | None = None,
    max_step: int | None = None,
    max_points: int = 200,
) -> str:
    try:
        api = _api()
        entity = _resolve_entity(api, entity)
        run = api.run(f"{entity}/{project}/{run_id}")

        max_points = min(max_points, 500)
        data_points = _fetch_metric_data(run, metrics, min_step, max_step, max_points)

        stats: dict[str, dict[str, Any]] = {}
        for m in metrics:
            values = [p[m] for p in data_points if p[m] is not None and isinstance(p[m], (int, float))]
            stats[m] = _metric_stats(values)

        return json.dumps({
            "run_id": run_id,
            "run_name": run.name,
            "num_points": len(data_points),
            "step_range": [data_points[0]["step"], data_points[-1]["step"]] if data_points else None,
            "stats": stats,
            "data": data_points,
        }, default=str)
    except Exception as e:
        return _error(f"Failed to get metrics for run '{run_id}': {e}")


# ---------------------------------------------------------------------------
# Tool 5: compare_runs
# ---------------------------------------------------------------------------

COMPARE_RUNS_DESC = """\
Compare 2 or more runs side-by-side on specific metrics.

Returns per-run summary stats, config diffs (only keys that differ between runs), \
and a combined data table with aligned steps for easy comparison.

Args:
  - project: W&B project name
  - run_ids: list of run IDs to compare (2-10 runs)
  - metrics: list of metric names to compare (e.g. ["train/loss", "eval/mae_999"])
  - entity: W&B entity/team (optional)
  - max_points: max data points per run (default 100)

This is the go-to tool for "which run was better?" questions.
"""


@mcp.tool(description=COMPARE_RUNS_DESC)
def compare_runs(
    project: str,
    run_ids: list[str],
    metrics: list[str],
    entity: str | None = None,
    max_points: int = 100,
) -> str:
    try:
        if len(run_ids) < 2:
            return _error("Need at least 2 run IDs to compare")
        if len(run_ids) > 10:
            return _error("Max 10 runs per comparison")

        api = _api()
        entity = _resolve_entity(api, entity)
        max_points = min(max_points, 200)

        runs_data = []
        all_configs: list[dict] = []

        for rid in run_ids:
            run = api.run(f"{entity}/{project}/{rid}")
            data_points = _fetch_metric_data(run, metrics, max_points=max_points)

            per_metric_stats: dict[str, dict] = {}
            for m in metrics:
                values = [p[m] for p in data_points if p[m] is not None and isinstance(p[m], (int, float))]
                per_metric_stats[m] = _metric_stats(values)

            runs_data.append({
                "run_id": rid,
                "run_name": run.name,
                "state": run.state,
                "created_at": run.created_at,
                "stats": per_metric_stats,
                "data": data_points,
            })
            all_configs.append({"run_id": rid, "run_name": run.name, "config": run.config})

        # Compute config diff — flatten nested dicts, skip per-run noise
        _NOISE_KEYS = {"run_name", "log_dir", "local_rank", "global_rank", "wandb_run_id"}

        def _flatten(d: dict, prefix: str = "") -> dict[str, Any]:
            out = {}
            for k, v in d.items():
                full = f"{prefix}{k}" if prefix else k
                if isinstance(v, dict):
                    out.update(_flatten(v, f"{full}."))
                else:
                    out[full] = v
            return out

        config_diff: dict[str, dict[str, Any]] = {}
        if len(all_configs) >= 2:
            flat_configs = [
                {"run_name": c["run_name"], "flat": _flatten(c["config"])}
                for c in all_configs
            ]
            all_keys: set[str] = set()
            for fc in flat_configs:
                all_keys.update(fc["flat"].keys())
            for key in sorted(all_keys):
                # Skip keys that are inherently per-run (not real hyperparams)
                leaf = key.rsplit(".", 1)[-1]
                if leaf in _NOISE_KEYS:
                    continue
                vals = [fc["flat"].get(key) for fc in flat_configs]
                if any(v != vals[0] for v in vals[1:]):
                    config_diff[key] = {
                        fc["run_name"]: fc["flat"].get(key) for fc in flat_configs
                    }

        # Build comparison summary table
        summary_table = []
        for m in metrics:
            row: dict[str, Any] = {"metric": m}
            for rd in runs_data:
                s = rd["stats"].get(m, {})
                row[rd["run_name"]] = {
                    "final": s.get("final"),
                    "min": s.get("min"),
                    "max": s.get("max"),
                    "mean": s.get("mean"),
                }
            summary_table.append(row)

        return json.dumps({
            "entity": entity,
            "project": project,
            "num_runs": len(runs_data),
            "metrics_compared": metrics,
            "summary": summary_table,
            "config_diff": config_diff if config_diff else "all configs identical",
            "runs": runs_data,
        }, default=str)
    except Exception as e:
        return _error(f"Failed to compare runs: {e}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
