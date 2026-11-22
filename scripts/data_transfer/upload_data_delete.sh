#!/usr/bin/env bash
set -euo pipefail

if [[ "${1:-}" != "--confirm-delete" ]]; then
  echo "This uploads data and deletes S3 files that are missing locally."
  echo "Run: $0 --confirm-delete [extra aws s3 sync args]"
  exit 2
fi
shift

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ARTIFACT_ROOT="${KUAFU_ARTIFACT_ROOT:-${PROJECT_ROOT}/../KuafuArtifacts}"
ENDPOINT_URL="https://s3.wasabisys.com"
PROFILE="kuafu"

aws s3 sync "$ARTIFACT_ROOT/Priors/" s3://kuafu/priors/ --delete --endpoint-url="$ENDPOINT_URL" --profile "$PROFILE" "$@"
aws s3 sync "$ARTIFACT_ROOT/Residuals/" s3://kuafu/residuals/ --delete --endpoint-url="$ENDPOINT_URL" --profile "$PROFILE" "$@"
aws s3 sync "$ARTIFACT_ROOT/Trajectories/" s3://kuafu/trajectories/ --delete --endpoint-url="$ENDPOINT_URL" --profile "$PROFILE" "$@"
