#!/usr/bin/env bash
# Push code to HiPerGator (rsync) and optionally to GitHub.
#   bash hpg/sync_push.sh [--with-env] [--dry-run] [--no-github]
# Datasets (*.npz) are excluded by default; copy them to ${HPG_REMOTE_DIR} once.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/env.sh"

WITH_ENV=false; DRY_RUN=false; SKIP_GITHUB=false
for arg in "$@"; do
  case "$arg" in
    --with-env) WITH_ENV=true ;;
    --dry-run) DRY_RUN=true ;;
    --no-github) SKIP_GITHUB=true ;;
    -h|--help) echo "Usage: $0 [--with-env] [--dry-run] [--no-github]"; exit 0 ;;
  esac
done

EXCLUDE_FILE="${SCRIPT_DIR}/sync_exclude.txt"
EXTRA_EXCLUDE=(); [[ "${WITH_ENV}" != "true" ]] && EXTRA_EXCLUDE=(--exclude '.env')
RSYNC_OPTS=(-avz --progress --exclude-from "${EXCLUDE_FILE}" "${EXTRA_EXCLUDE[@]}")
[[ "${DRY_RUN}" == "true" ]] && RSYNC_OPTS+=(--dry-run)

echo "=== Deploy MMCTL → HiPerGator ==="
echo "Local: ${MMCTL_LOCAL_ROOT}/"
echo "HPG:   ${HPG_SSH}:${HPG_REMOTE_DIR}/"

ssh -o ControlMaster=auto -o ControlPath=~/.ssh/cm-%r@%h:%p -o ControlPersist=600 \
  "${HPG_SSH}" "mkdir -p '${HPG_REMOTE_DIR}' '${HPG_REMOTE_DIR}/logs'"

echo "--- rsync → HiPerGator ---"
rsync "${RSYNC_OPTS[@]}" -e "${HPG_RSYNC_SSH}" \
  "${MMCTL_LOCAL_ROOT}/" "${HPG_SSH}:${HPG_REMOTE_DIR}/"

if [[ "${SKIP_GITHUB}" != "true" ]]; then
  branch="$(git -C "${MMCTL_LOCAL_ROOT}" rev-parse --abbrev-ref HEAD 2>/dev/null || echo '')"
  if [[ -n "${branch}" ]]; then
    echo "--- git push origin ${branch} ---"
    git -C "${MMCTL_LOCAL_ROOT}" push origin "${branch}" || echo "(git push skipped/failed)"
  fi
fi
echo "Done."
