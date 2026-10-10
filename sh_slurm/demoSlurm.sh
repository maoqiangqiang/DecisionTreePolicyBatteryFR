#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=0-23:55:00
#SBATCH --output=a_RADDT_PyDDPG_noInit_TrainandTest_%j_%N.out

cd "${SLURM_SUBMIT_DIR:-.}"

LOG_DIR="log"
mkdir -p "$LOG_DIR"

# activate virtual environment
xxxxxx

SEED=42
TREE_DEPTH=3
SCALE=20
SCALE_TAG="${SCALE//./p}"
JOB_ID_FOR_PORT="${SLURM_JOB_ID:-$$}"
export BATTERY_ZMQ_PORT=$((20000 + JOB_ID_FOR_PORT % 40000))

echo "BATTERY_ZMQ_PORT=${BATTERY_ZMQ_PORT}"

# Start Julia server in background
julia src/juliaBatteryServer.jl > "log/juliaServer_${SLURM_JOBID}.log" 2>&1 &

# Record Julia PID
JULIA_PID=$!
trap 'kill "$JULIA_PID" 2>/dev/null || true' EXIT

sleep 10

python test/test_RADDT_DDPG_RL_TrainandTest.py \
    --seed "$SEED" \
    --treeDepth "$TREE_DEPTH" \
    --scale "$SCALE" \
    > "log/RADDT_DDPGTrainandBatteryFRControlTest_seed${SEED}_D${TREE_DEPTH}_scale${SCALE_TAG}_${SLURM_JOBID}.log"
