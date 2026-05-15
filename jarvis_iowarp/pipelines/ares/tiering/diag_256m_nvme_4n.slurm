#!/bin/bash
#SBATCH --job-name=diag_256m_dram_nvme_4n
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=40
#SBATCH --partition=debug
#SBATCH --time=00:05:00
#SBATCH --output=/mnt/common/llogan/iowarp/core/slurm-%j.out
#SBATCH --error=/mnt/common/llogan/iowarp/core/slurm-%j.err

# Diagnostic wrapper for the 256m-block DRAM tiering pipeline.
# Runs `jarvis ppl run` in the background, polls the captured log,
# and when no log growth occurs for 90s with no "Max Write" line
# present, pstacks every chimaera / fuse_cte / ior PID on every
# node so we can root-cause the queue-drain hang.
#
# Diagnostics land in /mnt/common/llogan/cte_tiering_diag_<jobid>/.

set -uo pipefail

source /etc/profile.d/lmod.sh 2>/dev/null || source /etc/profile.d/modules.sh
module load gcc/11.4.0/openmpi/5.0.5-x2kvx5l ior/3.3.0-fihmxeb
mkdir -p /home/llogan/iowarp_prte_tmp

YAML=/home/llogan/iowarp/core/jarvis_iowarp/pipelines/ares/tiering/ior_cte_tiering_dram_nvme_4n.yaml
SHARED_DIR=/home/llogan/.ppi-jarvis/shared/ior_cte_tiering_dram_nvme_4n
HOSTFILE="$SHARED_DIR/hostfile.txt"
DIAG_DIR=/mnt/common/llogan/cte_tiering_diag_${SLURM_JOB_ID}
JARVIS_LOG="$DIAG_DIR/jarvis.log"

mkdir -p "$DIAG_DIR" "$(dirname "$HOSTFILE")"
scontrol show hostnames "$SLURM_JOB_NODELIST" | awk -v s=-40g '{print $0 s}' > "$HOSTFILE"
echo "Hostfile:" && cat "$HOSTFILE"

jarvis ppl run yaml "$YAML" > "$JARVIS_LOG" 2>&1 &
JARVIS_PID=$!
echo "Started jarvis pid=$JARVIS_PID"

LAST_SIZE=0
STALL=0
WROTE_OK=0
while kill -0 $JARVIS_PID 2>/dev/null; do
  sleep 30
  if grep -q "Max Write:" "$JARVIS_LOG" 2>/dev/null; then
    WROTE_OK=1
    break
  fi
  SIZE=$(stat -c%s "$JARVIS_LOG" 2>/dev/null || echo 0)
  if [ "$SIZE" = "$LAST_SIZE" ]; then
    STALL=$((STALL + 1))
    if [ "$STALL" -ge 3 ]; then
      echo "HUNG: no log growth for ~90s and no Max Write. Capturing pstacks..."
      break
    fi
  else
    STALL=0
  fi
  LAST_SIZE="$SIZE"
done

if [ "$WROTE_OK" = "1" ]; then
  echo "IOR write completed normally; waiting for jarvis to exit."
  wait $JARVIS_PID
  exit 0
fi

# Capture pstacks on every node in the SLURM allocation.
srun --jobid="$SLURM_JOB_ID" --ntasks-per-node=1 -N "$SLURM_JOB_NUM_NODES" \
  --output="$DIAG_DIR/pstack-%n.txt" --error="$DIAG_DIR/pstack-%n.err" \
  bash -c '
    set +e
    H=$(hostname)
    SELF_PID=$$
    echo "===== HOST $H pgrep listing ====="
    pgrep -au $USER -f "chimaera|wrp_cte|ior|jarvis|fuse" | head -200

    echo
    echo "===== HOST $H mounts ====="
    mount | grep -i "fuse\|cte_tiering" | head

    echo
    echo "===== HOST $H per-ior kernel stacks (all 24 ranks) ====="
    for pid in $(pgrep -u $USER -f "^ior "); do
      [ -d /proc/$pid ] || continue
      S=$(awk "/^State:/{print \$2\$3}" /proc/$pid/status)
      W=$(cat /proc/$pid/wchan 2>/dev/null)
      KSTACK=$(head -8 /proc/$pid/stack 2>/dev/null | tr "\n" "|")
      echo "pid=$pid state=$S wchan=$W kstack=[$KSTACK]"
    done

    echo
    echo "===== HOST $H full gdb backtraces (chimaera + wrp_cte_fuse + 2 ior) ====="
    CHIM_PIDS=$(pgrep -u $USER -f "^chimaera ")
    FUSE_PIDS=$(pgrep -u $USER -f "wrp_cte_fuse")
    IOR_PIDS=$(pgrep -u $USER -f "^ior " | head -2)
    for pid in $CHIM_PIDS $FUSE_PIDS $IOR_PIDS; do
      [ "$pid" = "$SELF_PID" ] && continue
      [ -d /proc/$pid ] || continue
      echo
      echo "===== pid=$pid host=$H cmd=$(tr -d \\0 < /proc/$pid/cmdline 2>/dev/null) ====="
      head -10 /proc/$pid/status 2>/dev/null
      gdb -batch -ex "set pagination off" -ex "thread apply all bt" -p "$pid" 2>&1 | head -600
    done
  '

echo "Diagnostics dumped to $DIAG_DIR"
kill -TERM $JARVIS_PID 2>/dev/null || true
sleep 5
kill -KILL $JARVIS_PID 2>/dev/null || true
wait $JARVIS_PID 2>/dev/null || true

# Try to clean up chimaera daemons on every node so the dependent
# slot is left in a sensible state for the next test.
srun --jobid="$SLURM_JOB_ID" --ntasks-per-node=1 -N "$SLURM_JOB_NUM_NODES" \
  bash -c 'pkill -KILL -u $USER -f "chimaera|fuse_cte" 2>/dev/null; fusermount3 -u $HOME/cte_tiering_dram_nvme_4n 2>/dev/null; true' \
  >/dev/null 2>&1 || true

exit 1
