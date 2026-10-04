#!/usr/bin/env bash
# Pull trained artifacts / results from HiPerGator back to the local repo.
#   bash hpg/sync_pull.sh [results|outputs|logs|all] [--dry-run]
# Default target: all (results + outputs).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/env.sh"

TARGET="${1:-all}"; shift || true
DRY_RUN=false
for arg in "$@"; do [[ "$arg" == "--dry-run" ]] && DRY_RUN=true; done

RSYNC_OPTS=(-avz --progress)
[[ "${DRY_RUN}" == "true" ]] && RSYNC_OPTS+=(--dry-run)

pull_path() {
  local sub="$1"
  if ! ssh -o ControlMaster=auto -o ControlPath=~/.ssh/cm-%r@%h:%p -o ControlPersist=600 \
        "${HPG_SSH}" "test -d '${HPG_REMOTE_DIR}/${sub}'" 2>/dev/null; then
    echo "  (skip — remote path missing: ${sub})"; return
  fi
  mkdir -p "${MMCTL_LOCAL_ROOT}/${sub}"
  echo "Pulling ${sub}/ -> local ${sub}/"
  rsync "${RSYNC_OPTS[@]}" -e "${HPG_RSYNC_SSH}" \
    "${HPG_SSH}:${HPG_REMOTE_DIR}/${sub}/" \
    "${MMCTL_LOCAL_ROOT}/${sub}/"
}

echo "=== Pull MMCTL artifacts from HiPerGator ==="
echo "HPG:   ${HPG_SSH}:${HPG_REMOTE_DIR}/"
echo "Local: ${MMCTL_LOCAL_ROOT}/"

case "${TARGET}" in
  results) pull_path "results" ;;
  outputs) pull_path "outputs" ;;
  logs)    pull_path "logs" ;;
  all)     pull_path "results"; pull_path "outputs" ;;
  *) echo "Unknown target '${TARGET}' (use: results|outputs|logs|all)"; exit 1 ;;
esac
echo "Done."
