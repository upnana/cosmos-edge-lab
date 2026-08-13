#!/usr/bin/env bash
# Sample GPU utilization / VRAM during train or eval.
#
# Usage:
#   bash scripts/monitor_gpu.sh wrap --tag vision_sft -- <command...>
#   bash scripts/monitor_gpu.sh start --tag action_train
#   bash scripts/monitor_gpu.sh stop  --tag action_train
#   bash scripts/monitor_gpu.sh summarize --csv path/to/gpu.csv
#
# Env:
#   GPU_MONITOR_INTERVAL  sample period seconds (default 5)
#   GPU_MONITOR_ROOT      output root (default $LAB_ROOT/outputs/gpu_monitor)
#   MONITOR_GPU           used by launch_*.sh (1=on, 0=off; default 1)
#
# Writes under GPU_MONITOR_ROOT/<tag>_<stamp>/:
#   gpu.csv       time series
#   summary.json  peak / mean util & memory
#   monitor.pid   sampler pid (while running)
#   meta.json     host / start info
set -euo pipefail

LAB_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
INTERVAL="${GPU_MONITOR_INTERVAL:-5}"
ROOT="${GPU_MONITOR_ROOT:-$LAB_ROOT/outputs/gpu_monitor}"

usage() {
  sed -n '2,22p' "$0" | sed 's/^# \?//'
  exit 2
}

_require_nvidia() {
  command -v nvidia-smi >/dev/null 2>&1 || {
    echo "ERROR: nvidia-smi not found" >&2
    exit 1
  }
}

_summarize_csv() {
  local csv="$1"
  local out="$2"
  python3 - "$csv" "$out" <<'PY'
import csv, json, sys, statistics
from pathlib import Path
csv_path, out_path = Path(sys.argv[1]), Path(sys.argv[2])
rows = []
if csv_path.exists():
    with csv_path.open(newline="") as f:
        for r in csv.DictReader(f):
            try:
                rows.append({
                    "gpu": int(float(r["gpu_index"])),
                    "util": float(r["utilization_gpu"]),
                    "mem": float(r["memory_used_miB"]),
                    "mem_total": float(r["memory_total_miB"]),
                    "ts": float(r["timestamp_unix"]),
                })
            except (KeyError, ValueError, TypeError):
                continue
summary = {"n_samples": len(rows), "gpus": {}, "csv": str(csv_path)}
if rows:
    summary["duration_s"] = round(max(r["ts"] for r in rows) - min(r["ts"] for r in rows), 1)
    by: dict[int, list] = {}
    for r in rows:
        by.setdefault(r["gpu"], []).append(r)
    for g, items in sorted(by.items()):
        utils = [x["util"] for x in items]
        mems = [x["mem"] for x in items]
        total = items[0]["mem_total"]
        summary["gpus"][str(g)] = {
            "name": None,
            "util_mean": round(statistics.fmean(utils), 2),
            "util_max": round(max(utils), 2),
            "util_p95": round(sorted(utils)[max(0, int(0.95 * (len(utils) - 1)))], 2),
            "mem_used_mean_miB": round(statistics.fmean(mems), 1),
            "mem_used_max_miB": round(max(mems), 1),
            "mem_total_miB": total,
            "mem_used_max_frac": round(max(mems) / total, 4) if total else None,
            "n": len(items),
        }
out_path.write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary, indent=2))
PY
}

cmd_start() {
  _require_nvidia
  local TAG="" OUT=""
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --tag) TAG="$2"; shift 2 ;;
      --out-dir) OUT="$2"; shift 2 ;;
      *) echo "ERROR: bad arg $1" >&2; exit 2 ;;
    esac
  done
  [[ -n "$TAG" ]] || { echo "ERROR: --tag required" >&2; exit 2; }
  mkdir -p "$ROOT"
  local stamp dir
  stamp=$(date +%Y%m%d_%H%M%S)
  dir="${OUT:-$ROOT/${TAG}_${stamp}}"
  mkdir -p "$dir"
  local csv="$dir/gpu.csv"
  local pidfile="$dir/monitor.pid"
  if [[ -f "$pidfile" ]] && kill -0 "$(cat "$pidfile")" 2>/dev/null; then
    echo "ERROR: monitor already running pid=$(cat "$pidfile") dir=$dir" >&2
    exit 1
  fi
  nvidia-smi --query-gpu=index,name,driver_version,memory.total --format=csv >"$dir/gpus.txt" || true
  cat >"$dir/meta.json" <<EOF
{
  "tag": "$TAG",
  "interval_s": $INTERVAL,
  "host": "$(hostname)",
  "started_at": "$(date -Is)",
  "csv": "$csv"
}
EOF
  # Detach sampler so `dir=$(cmd_start ...)` / callers never wait on it.
  nohup bash -c '
    csv="$1"; interval="$2"
    if [[ ! -f "$csv" ]]; then
      echo "timestamp_unix,timestamp_iso,gpu_index,name,utilization_gpu,utilization_memory,memory_used_miB,memory_total_miB,power_draw_w,temperature_c" >"$csv"
    fi
    trim() { local s="$1"; s="${s#"${s%%[![:space:]]*}"}"; s="${s%"${s##*[![:space:]]}"}"; printf "%s" "$s"; }
    while true; do
      unix=$(date +%s); ts=$(date -Is)
      nvidia-smi --query-gpu=index,name,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw,temperature.gpu \
        --format=csv,noheader,nounits 2>/dev/null \
        | while IFS= read -r line; do
            [[ -z "$line" ]] && continue
            IFS="," read -r idx name ug um mu mt pw temp <<<"$line"
            idx=$(trim "$idx"); name=$(trim "$name"); ug=$(trim "$ug"); um=$(trim "$um")
            mu=$(trim "$mu"); mt=$(trim "$mt"); pw=$(trim "$pw"); temp=$(trim "$temp")
            [[ "$pw" == *N/A* || "$pw" == *\[* ]] && pw=""
            printf "%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n" "$unix" "$ts" "$idx" "$name" "$ug" "$um" "$mu" "$mt" "$pw" "$temp"
          done >>"$csv" || true
      sleep "$interval"
    done
  ' bash "$csv" "$INTERVAL" >/dev/null 2>&1 &
  local pid=$!
  disown "$pid" 2>/dev/null || true
  echo "$pid" >"$pidfile"
  echo "$dir" >"$ROOT/.last_${TAG}.dir"
  echo "$dir" >"$ROOT/.last_monitor.dir"
  echo "$dir" >"$ROOT/${TAG}.active_dir"
  echo ">>> gpu monitor started tag=$TAG pid=$pid dir=$dir interval=${INTERVAL}s" >&2
  printf '%s\n' "$dir"
}

cmd_stop() {
  local TAG="" OUT=""
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --tag) TAG="$2"; shift 2 ;;
      --out-dir) OUT="$2"; shift 2 ;;
      *) echo "ERROR: bad arg $1" >&2; exit 2 ;;
    esac
  done
  [[ -n "$TAG" ]] || { echo "ERROR: --tag required" >&2; exit 2; }
  local dir="${OUT:-}"
  if [[ -z "$dir" && -f "$ROOT/${TAG}.active_dir" ]]; then
    dir=$(cat "$ROOT/${TAG}.active_dir")
  fi
  if [[ -z "$dir" && -f "$ROOT/.last_${TAG}.dir" ]]; then
    dir=$(cat "$ROOT/.last_${TAG}.dir")
  fi
  [[ -n "$dir" && -d "$dir" ]] || { echo "ERROR: no active monitor dir for tag=$TAG" >&2; exit 1; }
  local pidfile="$dir/monitor.pid"
  if [[ -f "$pidfile" ]]; then
    local pid
    pid=$(cat "$pidfile")
    if kill -0 "$pid" 2>/dev/null; then
      kill "$pid" 2>/dev/null || true
      wait "$pid" 2>/dev/null || true
    fi
    rm -f "$pidfile"
  fi
  rm -f "$ROOT/${TAG}.active_dir"
  echo ">>> gpu monitor stopped tag=$TAG dir=$dir" >&2
  _summarize_csv "$dir/gpu.csv" "$dir/summary.json"
  echo "  csv:     $dir/gpu.csv" >&2
  echo "  summary: $dir/summary.json" >&2
}

cmd_summarize() {
  local CSV=""
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --csv) CSV="$2"; shift 2 ;;
      *) echo "ERROR: bad arg $1" >&2; exit 2 ;;
    esac
  done
  [[ -n "$CSV" ]] || { echo "ERROR: --csv required" >&2; exit 2; }
  local tmp
  tmp=$(mktemp)
  _summarize_csv "$CSV" "$tmp"
  rm -f "$tmp"
}

cmd_wrap() {
  _require_nvidia
  local TAG="job" OUT="" 
  local -a REST=()
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --tag) TAG="$2"; shift 2 ;;
      --out-dir) OUT="$2"; shift 2 ;;
      --) shift; REST=("$@"); break ;;
      -h|--help) usage ;;
      *) REST=("$@"); break ;;
    esac
  done
  [[ ${#REST[@]} -gt 0 ]] || { echo "ERROR: wrap needs a command" >&2; exit 2; }

  local dir
  if [[ -n "$OUT" ]]; then
    dir=$(cmd_start --tag "$TAG" --out-dir "$OUT")
  else
    dir=$(cmd_start --tag "$TAG")
  fi

  local rc=0
  set +e
  "${REST[@]}"
  rc=$?
  set -e

  cmd_stop --tag "$TAG" --out-dir "$dir" >/dev/null || true
  # re-print summary for the log
  if [[ -f "$dir/summary.json" ]]; then
    echo ">>> GPU monitor summary ($TAG):" >&2
    cat "$dir/summary.json" >&2 || true
  fi
  return "$rc"
}

main() {
  [[ $# -ge 1 ]] || usage
  local sub="$1"; shift
  case "$sub" in
    start) cmd_start "$@" ;;
    stop) cmd_stop "$@" ;;
    summarize) cmd_summarize "$@" ;;
    wrap) cmd_wrap "$@" ;;
    -h|--help|help) usage ;;
    *) echo "ERROR: unknown subcommand $sub" >&2; usage ;;
  esac
}

main "$@"
