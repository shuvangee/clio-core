#!/bin/bash
#SBATCH --job-name=diag_256m_nvme_1n
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=40
#SBATCH --partition=debug
#SBATCH --time=00:05:00
#SBATCH --output=/mnt/common/llogan/iowarp/core/slurm-%j.out
#SBATCH --error=/mnt/common/llogan/iowarp/core/slurm-%j.err

set -uo pipefail
source /etc/profile.d/lmod.sh 2>/dev/null || source /etc/profile.d/modules.sh
module load gcc/11.4.0/openmpi/5.0.5-x2kvx5l ior/3.3.0-fihmxeb
mkdir -p /home/llogan/iowarp_prte_tmp

YAML=/home/llogan/iowarp/core/jarvis_iowarp/pipelines/ares/tiering/ior_cte_tiering_dram_nvme_1n.yaml
SHARED_DIR=/home/llogan/.ppi-jarvis/shared/ior_cte_tiering_dram_nvme_1n
HOSTFILE="$SHARED_DIR/hostfile.txt"
DIAG_DIR=/mnt/common/llogan/cte_tiering_diag_${SLURM_JOB_ID}
JARVIS_LOG="$DIAG_DIR/jarvis.log"

mkdir -p "$DIAG_DIR" "$(dirname "$HOSTFILE")"
scontrol show hostnames "$SLURM_JOB_NODELIST" | awk -v s=-40g '{print $0 s}' > "$HOSTFILE"

jarvis ppl run yaml "$YAML" > "$JARVIS_LOG" 2>&1 &
JARVIS_PID=$!

# Aggressive stall detection: 30s with no log growth -> capture
LAST_SIZE=0
STALL=0
WROTE_OK=0
for i in $(seq 1 12); do
  sleep 15
  if grep -q "Max Write:" "$JARVIS_LOG" 2>/dev/null; then WROTE_OK=1; break; fi
  SIZE=$(stat -c%s "$JARVIS_LOG" 2>/dev/null || echo 0)
  if [ "$SIZE" = "$LAST_SIZE" ]; then
    STALL=$((STALL + 1))
    if [ "$STALL" -ge 2 ]; then
      echo "STALL detected at $i*15s"
      break
    fi
  else STALL=0; fi
  LAST_SIZE="$SIZE"
done

if [ "$WROTE_OK" != "1" ]; then
  H=$(hostname)
  echo "===== pgrep listing ====="
  pgrep -au $USER -f "chimaera|wrp_cte|ior|jarvis|fuse" > "$DIAG_DIR/pgrep.txt"
  cat "$DIAG_DIR/pgrep.txt"

  echo "===== per-ior kernel stacks ====="
  for pid in $(pgrep -u $USER -f "^ior "); do
    [ -d /proc/$pid ] || continue
    S=$(awk "/^State:/{print \$2\$3}" /proc/$pid/status)
    W=$(cat /proc/$pid/wchan 2>/dev/null)
    echo "pid=$pid state=$S wchan=$W"
  done > "$DIAG_DIR/ior-states.txt"
  cat "$DIAG_DIR/ior-states.txt"

  echo "===== gdb backtraces ====="
  for pid in $(pgrep -u $USER -f "^chimaera ") $(pgrep -u $USER -f "wrp_cte_fuse") $(pgrep -u $USER -f "^ior " | head -2); do
    [ -d /proc/$pid ] || continue
    echo "===== pid=$pid cmd=$(tr -d \\0 < /proc/$pid/cmdline) =====" >> "$DIAG_DIR/gdb.txt"
    gdb -batch -ex "set pagination off" -ex "thread apply all bt" -p "$pid" >> "$DIAG_DIR/gdb.txt" 2>&1
  done
fi

kill -KILL $JARVIS_PID 2>/dev/null
sleep 2
pkill -KILL -u $USER -f "chimaera|wrp_cte|^ior " 2>/dev/null
fusermount3 -u $HOME/cte_tiering_dram_nvme_1n 2>/dev/null
exit 0
