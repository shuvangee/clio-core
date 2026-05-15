#!/bin/bash
#SBATCH --job-name=ping_test
#SBATCH --partition=compute
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:02:00
#SBATCH --output=/home/llogan/iowarp/core/slurm-%j.out
#SBATCH --error=/home/llogan/iowarp/core/slurm-%j.err
#SBATCH --nodelist=ares-comp-14,ares-comp-18

# Test 40 GbE reachability of ares-comp-18 from another node.
NODES=( $(scontrol show hostnames "$SLURM_JOB_NODELIST") )
SRC=${NODES[0]}; DST=${NODES[1]}
echo "SRC=$SRC DST=$DST"
echo "=== ping 40g IP ==="
ssh -o StrictHostKeyChecking=no "$SRC" "ping -c 3 -W 2 172.25.101.18"
echo "=== iperf3 quick ==="
ssh -o StrictHostKeyChecking=no "$DST" "iperf3 -s -p 15301 -1 -i 0" > /tmp/iperf3_dst.log 2>&1 &
SPID=$!
sleep 1
ssh -o StrictHostKeyChecking=no "$SRC" "iperf3 -c $DST-40g -p 15301 -t 5 -l 1M" || true
wait $SPID 2>/dev/null || true
echo "=== iperf3 done ==="
