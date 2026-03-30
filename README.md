# wb-runs-mcp

A minimal, composable MCP server for Weights & Biases. Built for LLMs that need to read experiment data without wrestling with GraphQL.

## Why?

The official W&B MCP server exposes raw GraphQL and expects the LLM to construct valid queries with pagination, filter escaping, and connection patterns. It fails constantly.

This server has 5 tools that use the W&B Python SDK directly:

| Tool | What it does |
|---|---|
| `list_projects` | Discover available projects |
| `list_runs` | Search/filter runs with regex, state, date, config, tags |
| `get_run` | Full detail: config, metric names, summary values, step count |
| `get_metrics` | Time-series data with auto-downsampling + summary stats |
| `compare_runs` | Side-by-side comparison with config diffs and aligned metrics |

## Setup

### 1. Install

```bash
# Using uv (recommended)
uv pip install wb-runs-mcp

# Or from source
git clone https://github.com/your-org/wb-runs-mcp
cd wb-runs-mcp
uv venv && uv pip install -e .
```

### 2. Set your API key

```bash
export WANDB_API_KEY=your_key_here
# Get one at https://wandb.ai/authorize
```

### 3. Add to Claude Code

Add to your `.mcp.json` (project-level) or `~/.mcp.json` (global):

```json
{
  "mcpServers": {
    "wb": {
      "command": "wb-runs-mcp",
      "env": {
        "WANDB_API_KEY": "your_key_here"
      }
    }
  }
}
```

Or if running from source:

```json
{
  "mcpServers": {
    "wb": {
      "command": "uv",
      "args": ["--directory", "/path/to/wb-runs-mcp", "run", "wb-runs-mcp"],
      "env": {
        "WANDB_API_KEY": "your_key_here"
      }
    }
  }
}
```

## Usage examples

Once connected, your LLM can:

**"What projects do I have?"**
→ Calls `list_projects()`

**"Show me the latest training run"**
→ Calls `list_runs(project="my-project", limit=1)`

**"What's currently running?"**
→ Calls `list_runs(project="my-project", state="running")`

**"How did the loss look for run m94p3szz?"**
→ Calls `get_metrics(project="my-project", run_id="m94p3szz", metrics=["train/loss"])`

**"Compare the last two runs on loss and accuracy"**
→ Calls `compare_runs(project="my-project", run_ids=["abc", "xyz"], metrics=["train/loss", "eval/accuracy"])`

**"Zoom into steps 1000-2000 of the loss curve"**
→ Calls `get_metrics(..., min_step=1000, max_step=2000)`

## Design

- **No GraphQL** — uses the W&B Python SDK directly
- **Auto-downsampling** — never returns more than 200-500 data points (configurable)
- **Summary stats** — min/max/mean/final computed server-side
- **Config diffs** — `compare_runs` shows only config keys that differ
- **Clean errors** — structured JSON errors, not Python tracebacks
- **~300 lines** — easy to audit, fork, and extend

## Tools reference

### list_projects

```
list_projects(entity?: string)
```

Lists all projects. If `entity` is omitted, returns projects for your account and all teams.

### list_runs

```
list_runs(
  project: string,
  entity?: string,
  name_contains?: string,      # case-insensitive substring
  name_regex?: string,          # regex pattern
  state?: string,               # "finished" | "running" | "crashed" | "failed"
  created_after?: string,       # ISO date
  created_before?: string,      # ISO date
  config_filters?: object,      # e.g. {"lr": 0.001}
  tags?: string[],              # all must match
  include_config?: boolean,     # include full config per run
  limit?: number,               # default 10, max 50
  offset?: number               # for pagination
)
```

### get_run

```
get_run(project: string, run_id: string, entity?: string)
```

Returns config, available metrics (grouped by prefix), summary values, step count.

### get_metrics

```
get_metrics(
  project: string,
  run_id: string,
  metrics: string[],            # e.g. ["train/loss", "eval/mae"]
  entity?: string,
  min_step?: number,
  max_step?: number,
  max_points?: number           # default 200, max 500
)
```

Returns `{step, metric_name: value}` data points plus `{min, max, mean, final, count}` stats per metric.

### compare_runs

```
compare_runs(
  project: string,
  run_ids: string[],            # 2-10 run IDs
  metrics: string[],
  entity?: string,
  max_points?: number           # default 100, max 200
)
```

Returns summary table, config diff (only keys that differ), and per-run data.

## License

MIT
