#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ARTIFACT_ROOT="${KUAFU_ARTIFACT_ROOT:-${PROJECT_ROOT}/../KuafuArtifacts}"
ENDPOINT_URL="https://s3.wasabisys.com"
PROFILE="kuafu"

aws s3 sync "$ARTIFACT_ROOT/Priors/" s3://kuafu/priors/ --endpoint-url="$ENDPOINT_URL" --profile "$PROFILE"
aws s3 sync "$ARTIFACT_ROOT/Residuals/" s3://kuafu/residuals/ --endpoint-url="$ENDPOINT_URL" --profile "$PROFILE"
aws s3 sync "$ARTIFACT_ROOT/Trajectories/" s3://kuafu/trajectories/ --endpoint-url="$ENDPOINT_URL" --profile "$PROFILE"
