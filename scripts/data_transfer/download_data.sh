#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ARTIFACT_ROOT="${KUAFU_ARTIFACT_ROOT:-${PROJECT_ROOT}/../KuafuArtifacts}"
ENDPOINT_URL="https://s3.wasabisys.com"
PROFILE="kuafu"

aws s3 sync s3://kuafu/priors/ "$ARTIFACT_ROOT/Priors/" --endpoint-url="$ENDPOINT_URL" --profile "$PROFILE"
aws s3 sync s3://kuafu/residuals/ "$ARTIFACT_ROOT/Residuals/" --endpoint-url="$ENDPOINT_URL" --profile "$PROFILE"
aws s3 sync s3://kuafu/trajectories/ "$ARTIFACT_ROOT/Trajectories/" --endpoint-url="$ENDPOINT_URL" --profile "$PROFILE"
