#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ARTIFACT_ROOT="${KUAFU_ARTIFACT_ROOT:-${PROJECT_ROOT}/../KuafuArtifacts}"

ENDPOINT_URL="${ENDPOINT_URL:-https://s3.wasabisys.com}"
PROFILE="${PROFILE:-kuafu}"
S3_BASE="${S3_BASE:-s3://kuafu/residuals}"
S3_PRIORS_BASE="${S3_PRIORS_BASE:-s3://kuafu/priors}"
DEST_ROOT="${DEST_ROOT:-${ARTIFACT_ROOT}/Residuals}"
PRIORS_DEST_ROOT="${PRIORS_DEST_ROOT:-${ARTIFACT_ROOT}/Priors}"
CSV="$PROJECT_ROOT/resources/checkpoint_catalogs/server_residual_success_epoch_ema_best_by_task.csv"
PRIORS_CSV=""
PEN_CUP_TASK="FR3-FinRay-Table-Pen-Cup-ff"
PEN_CUP_BEST_CHECKPOINT="${PEN_CUP_BEST_CHECKPOINT:-FR3-FinRay-Table-Pen-Cup-ff/2026-05-23_17-55-28/rl_runs/2026-05-23_17-57-45_Run-2/nn/last_pen-cup_ep_1450_rew_14853.837.pth}"
PEN_CUP_BEST_JSON="${PEN_CUP_BEST_JSON:-FR3-FinRay-Table-Pen-Cup-ff/2026-05-23_17-52-24/best.json}"
HANG_CUP_FFF_TASK="FR3-FinRay-Table-Hang-Cup-fff"
HANG_CUP_FFF_BEST_CHECKPOINT="${HANG_CUP_FFF_BEST_CHECKPOINT:-FR3-FinRay-Table-Hang-Cup-fff/2026-05-22_17-09-20/rl_runs/2026-05-22_17-11-25_Run-0/nn/last_fr3-finray-hang-cup_ep_3500_rew_13644.76.pth}"
HANG_CUP_FFF_BEST_JSON="${HANG_CUP_FFF_BEST_JSON:-FR3-FinRay-Table-Hang-Cup-fff/2026-05-22_17-07-48/best.json}"
HANG_CUP_FFF_PRIOR_CHECKPOINT="${HANG_CUP_FFF_PRIOR_CHECKPOINT:-Priors/herm/hang_cup/hang_cup_chunking_10_state_fff/best.pth}"
CAN_TASK="FR3-FinRay-Table-Can-ff"
CAN_BEST_CHECKPOINT="${CAN_BEST_CHECKPOINT:-FR3-FinRay-Table-Can-ff/2026-05-25_17-02-29/rl_runs/2026-05-25_17-04-45_Run-1/nn/last_fr3-finray-can_ep_1800_rew_10335.244.pth}"
CAN_BEST_JSON="${CAN_BEST_JSON:-FR3-FinRay-Table-Can-ff/2026-05-25_16-59-11/best.json}"
ESTOP_OPEN_TASK="FR3-FinRay-Table-EStop-Open-ff"
ESTOP_OPEN_BEST_CHECKPOINT="${ESTOP_OPEN_BEST_CHECKPOINT:-FR3-FinRay-Table-EStop-Open-ff/2026-05-25_19-11-40/rl_runs/2026-05-25_19-12-48_Run-0/nn/last_Open-FR3FinRay-EStop_ep_1000_rew_1675.231.pth}"
DRY_RUN=0
PRINT_PLAN=0
ALLOW_MISSING=0

usage() {
  cat <<'EOF'
Usage:
  scripts/data_transfer/download_success_epoch_ema_checkpoints.sh [options] [checkpoint_report.csv]

Downloads only the residual checkpoints listed in the CSV's "checkpoint"
column, YAML/JSON settings files from the corresponding residual run groups,
initial_weights_params JSON files referenced by downloaded run_params.json files,
and prior checkpoints listed in the matching prior CSV.

Defaults:
  CSV:       resources/checkpoint_catalogs/server_residual_success_epoch_ema_best_by_task.csv
  Prior CSV: Same path as CSV with _priors.csv suffix.
  Output:    $KUAFU_ARTIFACT_ROOT/Residuals/
  Priors:    $KUAFU_ARTIFACT_ROOT/Priors/
  S3 source: s3://kuafu/residuals/
  Prior S3:  s3://kuafu/priors/

Options:
  --dest DIR         Local Residuals destination root.
  --priors-dest DIR  Local Priors destination root.
  --priors-csv FILE  Prior checkpoint CSV. Default: input CSV with _priors.csv suffix.
  --profile NAME     AWS profile name. Default: kuafu
  --endpoint URL     S3 endpoint URL. Default: https://s3.wasabisys.com
  --dry-run          Ask aws CLI to print what it would copy.
  --print-plan       Print checkpoint/settings download plan and exit.
  --allow-missing    Continue successfully when listed residual checkpoints are absent.
  -h, --help         Show this help.

Examples:
  scripts/data_transfer/download_success_epoch_ema_checkpoints.sh
  scripts/data_transfer/download_success_epoch_ema_checkpoints.sh resources/checkpoint_catalogs/server_residual_success_epoch_ema_checkpoints.csv
  scripts/data_transfer/download_success_epoch_ema_checkpoints.sh --priors-csv resources/checkpoint_catalogs/server_residual_success_epoch_ema_checkpoints_priors.csv resources/checkpoint_catalogs/server_residual_success_epoch_ema_checkpoints.csv
  scripts/data_transfer/download_success_epoch_ema_checkpoints.sh --dest /tmp/Residuals --print-plan
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dest)
      DEST_ROOT="$2"
      shift 2
      ;;
    --priors-dest)
      PRIORS_DEST_ROOT="$2"
      shift 2
      ;;
    --priors-csv)
      PRIORS_CSV="$2"
      shift 2
      ;;
    --profile)
      PROFILE="$2"
      shift 2
      ;;
    --endpoint)
      ENDPOINT_URL="$2"
      shift 2
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    --print-plan)
      PRINT_PLAN=1
      shift
      ;;
    --allow-missing)
      ALLOW_MISSING=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    -*)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
    *)
      CSV="$1"
      shift
      ;;
  esac
done

if [[ "$CSV" != /* ]]; then
  CSV="$PROJECT_ROOT/$CSV"
fi

if [[ ! -f "$CSV" ]]; then
  echo "CSV not found: $CSV" >&2
  exit 1
fi

if [[ -z "$PRIORS_CSV" ]]; then
  if [[ "$CSV" == *.csv ]]; then
    PRIORS_CSV="${CSV%.csv}_priors.csv"
  else
    PRIORS_CSV="$CSV.priors.csv"
  fi
elif [[ "$PRIORS_CSV" != /* ]]; then
  PRIORS_CSV="$PROJECT_ROOT/$PRIORS_CSV"
fi

if [[ ! -f "$PRIORS_CSV" ]]; then
  echo "Prior CSV not found: $PRIORS_CSV" >&2
  echo "Pass --priors-csv FILE, or create the default matching prior CSV." >&2
  exit 1
fi

TMP_DIR="$(mktemp -d)"
trap 'rm -rf "$TMP_DIR"' EXIT

python3 - "$CSV" "$TMP_DIR/checkpoints.txt" "$TMP_DIR/settings_prefixes.txt" \
  "$PEN_CUP_TASK" "$PEN_CUP_BEST_CHECKPOINT" \
  "$HANG_CUP_FFF_TASK" "$HANG_CUP_FFF_BEST_CHECKPOINT" \
  "$CAN_TASK" "$CAN_BEST_CHECKPOINT" \
  "$ESTOP_OPEN_TASK" "$ESTOP_OPEN_BEST_CHECKPOINT" <<'PY'
import csv
import sys
from pathlib import Path

csv_path = Path(sys.argv[1])
checkpoint_out = Path(sys.argv[2])
prefix_out = Path(sys.argv[3])
pen_cup_task = sys.argv[4].strip().rstrip("/")
pen_cup_best_checkpoint_raw = sys.argv[5].strip()
hang_cup_fff_task = sys.argv[6].strip().rstrip("/")
hang_cup_fff_best_checkpoint_raw = sys.argv[7].strip()
can_task = sys.argv[8].strip().rstrip("/")
can_best_checkpoint_raw = sys.argv[9].strip()
estop_open_task = sys.argv[10].strip().rstrip("/")
estop_open_best_checkpoint_raw = sys.argv[11].strip()


def normalize_checkpoint(path: str) -> str:
    path = path.strip().replace("\\", "/")
    for marker in ("s3://kuafu/residuals/", "residuals/", "Residuals/"):
        if path.startswith(marker):
            path = path[len(marker):]
    path = path.lstrip("/")
    if not path or path.startswith("../") or "/../" in path:
        raise ValueError(f"unsafe checkpoint path: {path!r}")
    return path


checkpoints: list[str] = []
pen_cup_best_checkpoint = normalize_checkpoint(pen_cup_best_checkpoint_raw)
hang_cup_fff_best_checkpoint = normalize_checkpoint(hang_cup_fff_best_checkpoint_raw)
can_best_checkpoint = normalize_checkpoint(can_best_checkpoint_raw)
estop_open_best_checkpoint = normalize_checkpoint(estop_open_best_checkpoint_raw)
with csv_path.open(newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    if not reader.fieldnames or "checkpoint" not in reader.fieldnames:
        raise SystemExit(f"{csv_path} does not have a 'checkpoint' column")
    for row in reader:
        checkpoint = normalize_checkpoint(row.get("checkpoint", ""))
        if checkpoint.startswith(pen_cup_task + "/"):
            checkpoint = pen_cup_best_checkpoint
        if checkpoint.startswith(hang_cup_fff_task + "/"):
            checkpoint = hang_cup_fff_best_checkpoint
        if checkpoint.startswith(can_task + "/"):
            checkpoint = can_best_checkpoint
        if checkpoint.startswith(estop_open_task + "/"):
            checkpoint = estop_open_best_checkpoint
        if not checkpoint.endswith(".pth"):
            raise SystemExit(f"checkpoint path does not end with .pth: {checkpoint}")
        checkpoints.append(checkpoint)

# Keep pinned picks available even if a caller passes an older CSV without the
# task rows.
if pen_cup_best_checkpoint:
    checkpoints.append(pen_cup_best_checkpoint)
if hang_cup_fff_best_checkpoint:
    checkpoints.append(hang_cup_fff_best_checkpoint)
if can_best_checkpoint:
    checkpoints.append(can_best_checkpoint)
if estop_open_best_checkpoint:
    checkpoints.append(estop_open_best_checkpoint)

checkpoints = sorted(dict.fromkeys(checkpoints))

settings_prefixes: set[str] = set()
for checkpoint in checkpoints:
    parts = checkpoint.split("/")
    # Current residual layout is Task/collection_timestamp/rl_runs/.../nn/file.pth.
    # Pull metadata from the selected collection root so JSON/YAML layouts remain identical.
    if len(parts) >= 2:
        settings_prefixes.add("/".join(parts[:2]))
    else:
        settings_prefixes.add(str(Path(checkpoint).parent))

checkpoint_out.write_text("\n".join(checkpoints) + ("\n" if checkpoints else ""), encoding="utf-8")
prefix_out.write_text("\n".join(sorted(settings_prefixes)) + ("\n" if settings_prefixes else ""), encoding="utf-8")
PY

python3 - "$PRIORS_CSV" "$TMP_DIR/prior_checkpoints.txt" "$HANG_CUP_FFF_PRIOR_CHECKPOINT" <<'PY'
import csv
import sys
from pathlib import Path

csv_path = Path(sys.argv[1])
prior_out = Path(sys.argv[2])
pinned_priors_raw = sys.argv[3:]


def normalize_prior(path: str) -> str:
    path = path.strip().replace("\\", "/")
    for marker in ("s3://kuafu/priors/", "priors/"):
        if path.startswith(marker):
            path = path[len(marker):]
            break
    path = path.lstrip("/")
    if not path.startswith("Priors/"):
        path = "Priors/" + path
    if path == "Priors/" or path.startswith("Priors/../") or "/../" in path:
        raise ValueError(f"unsafe prior checkpoint path: {path!r}")
    return path


priors: list[str] = []
with csv_path.open(newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    if not reader.fieldnames:
        raise SystemExit(f"{csv_path} has no CSV header")
    column = "prior_checkpoint" if "prior_checkpoint" in reader.fieldnames else "checkpoint"
    if column not in reader.fieldnames:
        raise SystemExit(f"{csv_path} must have a 'prior_checkpoint' or 'checkpoint' column")
    for row in reader:
        prior = normalize_prior(row.get(column, ""))
        if not prior.endswith(".pth"):
            raise SystemExit(f"prior checkpoint path does not end with .pth: {prior}")
        priors.append(prior)

for raw_prior in pinned_priors_raw:
    prior = normalize_prior(raw_prior)
    if not prior.endswith(".pth"):
        raise SystemExit(f"prior checkpoint path does not end with .pth: {prior}")
    priors.append(prior)

priors = sorted(dict.fromkeys(priors))
prior_out.write_text("\n".join(priors) + ("\n" if priors else ""), encoding="utf-8")
PY

mapfile -t CHECKPOINTS < "$TMP_DIR/checkpoints.txt"
mapfile -t SETTINGS_PREFIXES < "$TMP_DIR/settings_prefixes.txt"
mapfile -t PRIOR_CHECKPOINTS < "$TMP_DIR/prior_checkpoints.txt"

echo "CSV: $CSV"
echo "Prior CSV: $PRIORS_CSV"
echo "Destination: $DEST_ROOT"
echo "Priors destination: $PRIORS_DEST_ROOT"
echo "Checkpoint files: ${#CHECKPOINTS[@]}"
echo "Settings prefixes: ${#SETTINGS_PREFIXES[@]}"
echo "Prior checkpoint files: ${#PRIOR_CHECKPOINTS[@]}"

if [[ "$PRINT_PLAN" -eq 1 ]]; then
  echo
  echo "Pinned Pen-Cup residual:"
  echo "  checkpoint: $PEN_CUP_BEST_CHECKPOINT"
  echo "  best.json:  $PEN_CUP_BEST_JSON"
  echo
  echo "Pinned Hang-Cup fff residual:"
  echo "  checkpoint: $HANG_CUP_FFF_BEST_CHECKPOINT"
  echo "  best.json:  $HANG_CUP_FFF_BEST_JSON"
  echo "  prior:      $HANG_CUP_FFF_PRIOR_CHECKPOINT"
  echo
  echo "Pinned Can residual:"
  echo "  checkpoint: $CAN_BEST_CHECKPOINT"
  echo "  best.json:  $CAN_BEST_JSON"
  echo
  echo "Pinned EStop Open residual:"
  echo "  checkpoint: $ESTOP_OPEN_BEST_CHECKPOINT"
  echo
  echo "Residual checkpoints:"
  printf '  %s\n' "${CHECKPOINTS[@]}"
  echo
  echo "Settings prefixes:"
  printf '  %s\n' "${SETTINGS_PREFIXES[@]}"
  echo
  echo "Prior checkpoints:"
  if [[ "${#PRIOR_CHECKPOINTS[@]}" -gt 0 ]]; then
    printf '  %s\n' "${PRIOR_CHECKPOINTS[@]}"
  else
    echo "  none"
  fi
  exit 0
fi

mkdir -p "$DEST_ROOT"
mkdir -p "$PRIORS_DEST_ROOT"

AWS_COMMON=(
  --endpoint-url="$ENDPOINT_URL"
  --profile "$PROFILE"
  --only-show-errors
)

if [[ "$DRY_RUN" -eq 1 ]]; then
  AWS_COMMON+=(--dryrun)
fi

MISSING_RESIDUAL_CHECKPOINTS=()

prior_s3_candidates() {
  local prior_key="$1"

  printf '%s\n' "$prior_key"

  # These stack2 priors are referenced by training settings under Priors/thor,
  # but the Wasabi objects currently live in the older archive prefix.
  if [[ "$prior_key" == thor/stack_2_damped_higher_noise_state/* ]]; then
    printf 'potential_useful/%s\n' "${prior_key#thor/}"
  fi
}

copy_prior_checkpoint() {
  local checkpoint="$1"
  local prior_key="${checkpoint#Priors/}"
  local dest="$PRIORS_DEST_ROOT/$prior_key"
  local candidate
  local output
  local -a tried=()

  mkdir -p "$(dirname "$dest")"
  if [[ -s "$dest" && "$DRY_RUN" -eq 0 ]]; then
    echo "Already exists: $dest"
    return 0
  fi

  while IFS= read -r candidate; do
    [[ -z "$candidate" ]] && continue
    tried+=("$candidate")

    if output="$(aws s3 cp "$S3_PRIORS_BASE/$candidate" "$dest" "${AWS_COMMON[@]}" 2>&1)"; then
      if [[ -n "$output" ]]; then
        printf '%s\n' "$output"
      fi
      if [[ "$candidate" != "$prior_key" ]]; then
        echo "Fetched $checkpoint from $S3_PRIORS_BASE/$candidate"
      fi
      return 0
    fi
  done < <(prior_s3_candidates "$prior_key")

  echo "Failed to download prior checkpoint: $checkpoint" >&2
  echo "Tried:" >&2
  for candidate in "${tried[@]}"; do
    echo "  $S3_PRIORS_BASE/$candidate" >&2
  done
  return 1
}

copy_residual_checkpoint() {
  local checkpoint="$1"
  local dest="$DEST_ROOT/$checkpoint"
  local output

  mkdir -p "$(dirname "$dest")"
  if [[ -s "$dest" && "$DRY_RUN" -eq 0 ]]; then
    echo "Already exists: $dest"
    return 0
  fi

  if output="$(aws s3 cp "$S3_BASE/$checkpoint" "$dest" "${AWS_COMMON[@]}" 2>&1)"; then
    if [[ -n "$output" ]]; then
      printf '%s\n' "$output"
    fi
    return 0
  fi

  echo "Missing listed residual checkpoint: $checkpoint" >&2
  MISSING_RESIDUAL_CHECKPOINTS+=("$checkpoint")
  return 0
}

echo
echo "Downloading YAML/JSON settings for selected residual groups..."
for prefix in "${SETTINGS_PREFIXES[@]}"; do
  mkdir -p "$DEST_ROOT/$prefix"
  aws s3 cp \
    "$S3_BASE/$prefix/" \
    "$DEST_ROOT/$prefix/" \
    --recursive \
    --exclude "*" \
    --include "*.yaml" \
    --include "*.yml" \
    --include "*.json" \
    --include "*.jsonl" \
    "${AWS_COMMON[@]}"
done

python3 - "$DEST_ROOT" "$TMP_DIR/settings_prefixes.txt" "$TMP_DIR/initial_weights_json.txt" \
  "$PEN_CUP_BEST_JSON" "$HANG_CUP_FFF_BEST_JSON" "$CAN_BEST_JSON" <<'PY'
import json
import shlex
import sys
from pathlib import Path

dest_root = Path(sys.argv[1])
prefixes_path = Path(sys.argv[2])
output_path = Path(sys.argv[3])
pinned_json_raw = [arg.strip() for arg in sys.argv[4:]]


def normalize_residual_json(path: str) -> str | None:
    path = path.strip().replace("\\", "/")
    for marker in (
        "s3://kuafu/residuals/",
        "/Residuals/",
        "Residuals/",
        "/logs/residual_rl/",
        "logs/residual_rl/",
    ):
        if marker in path:
            path = path.split(marker, 1)[1]
            break
    path = path.lstrip("/")
    if not path.endswith(".json") or not path or path.startswith("../") or "/../" in path:
        return None
    return path


def initial_weights_args(run_params: dict) -> list[str]:
    argv = run_params.get("argv") or []
    candidates: list[str] = []
    for args in (argv, shlex.split(run_params.get("command_line", ""))):
        for i, arg in enumerate(args):
            if arg == "--initial_weights_params" and i + 1 < len(args):
                candidates.append(args[i + 1])
    return candidates


weights_json: set[str] = set()
for raw_json in pinned_json_raw:
    pinned_json = normalize_residual_json(raw_json)
    if pinned_json:
        weights_json.add(pinned_json)
prefixes = [line.strip() for line in prefixes_path.read_text(encoding="utf-8").splitlines() if line.strip()]
for prefix in prefixes:
    prefix_dir = dest_root / prefix
    for run_params_path in prefix_dir.rglob("run_params.json"):
        try:
            run_params = json.loads(run_params_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        for raw_path in initial_weights_args(run_params):
            normalized = normalize_residual_json(raw_path)
            if normalized:
                weights_json.add(normalized)

output_path.write_text(
    "\n".join(sorted(weights_json)) + ("\n" if weights_json else ""),
    encoding="utf-8",
)
PY

mapfile -t INITIAL_WEIGHTS_JSON < "$TMP_DIR/initial_weights_json.txt"

echo
echo "Downloading referenced initial weights JSON files..."
if [[ "${#INITIAL_WEIGHTS_JSON[@]}" -eq 0 ]]; then
  echo "No initial weights JSON files were referenced by downloaded run_params.json files."
else
  for json_file in "${INITIAL_WEIGHTS_JSON[@]}"; do
    mkdir -p "$(dirname "$DEST_ROOT/$json_file")"
    aws s3 cp \
      "$S3_BASE/$json_file" \
      "$DEST_ROOT/$json_file" \
      "${AWS_COMMON[@]}"
  done
fi

echo
echo "Downloading referenced prior checkpoints..."
if [[ "${#PRIOR_CHECKPOINTS[@]}" -eq 0 ]]; then
  echo "No prior checkpoints were referenced by the prior CSV."
else
  for checkpoint in "${PRIOR_CHECKPOINTS[@]}"; do
    copy_prior_checkpoint "$checkpoint"
  done
fi

echo
echo "Downloading listed residual checkpoints..."
for checkpoint in "${CHECKPOINTS[@]}"; do
  copy_residual_checkpoint "$checkpoint"
done

if [[ "${#MISSING_RESIDUAL_CHECKPOINTS[@]}" -gt 0 ]]; then
  echo >&2
  echo "Missing residual checkpoints:" >&2
  printf '  %s\n' "${MISSING_RESIDUAL_CHECKPOINTS[@]}" >&2
  if [[ "$ALLOW_MISSING" -eq 0 ]]; then
    echo "Upload the missing objects, remove those rows from the CSV, or rerun with --allow-missing to accept a partial download." >&2
    exit 1
  fi
fi

echo
echo "Done."
