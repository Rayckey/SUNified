#!/usr/bin/env bash
set -euo pipefail

if [[ "${1:-}" != "--confirm-delete" ]]; then
  echo "This downloads data and deletes local files that are missing from S3."
  echo "Run: $0 --confirm-delete [extra aws s3 sync args]"
  exit 2
fi
shift

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ARTIFACT_ROOT="${KUAFU_ARTIFACT_ROOT:-${PROJECT_ROOT}/../KuafuArtifacts}"
ENDPOINT_URL="https://s3.wasabisys.com"
PROFILE="kuafu"

aws s3 sync s3://kuafu/priors/ "$ARTIFACT_ROOT/Priors/" --delete --endpoint-url="$ENDPOINT_URL" --profile "$PROFILE" "$@"
aws s3 sync s3://kuafu/residuals/ "$ARTIFACT_ROOT/Residuals/" --delete --endpoint-url="$ENDPOINT_URL" --profile "$PROFILE" "$@"
aws s3 sync s3://kuafu/trajectories/ "$ARTIFACT_ROOT/Trajectories/" --delete --endpoint-url="$ENDPOINT_URL" --profile "$PROFILE" "$@"
