#!/usr/bin/env bash
# Source this file: source hpg/env.sh
# Override any variable before sourcing for your machine.

export MMCTL_LOCAL_ROOT="${MMCTL_LOCAL_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"

export HPG_USER="${HPG_USER:-zhou.m}"
export HPG_HOST="${HPG_HOST:-hpg.rc.ufl.edu}"
export HPG_REMOTE_DIR="${HPG_REMOTE_DIR:-/blue/azare/zhou.m/MMCTL}"

export HPG_SSH="${HPG_USER}@${HPG_HOST}"
export HPG_RSYNC_SSH="ssh -o ControlMaster=auto -o ControlPath=~/.ssh/cm-%r@%h:%p -o ControlPersist=600"

# sbatch --export: propagate user env but exclude vars that break the project venv.
# Names listed after ALL are NOT propagated into the job (see `man sbatch`).
export SLURM_EXPORT_BASE="ALL,PYTHONPATH,PYTHONHOME,PYTHONUSERBASE,PYTHONSTARTUP"
