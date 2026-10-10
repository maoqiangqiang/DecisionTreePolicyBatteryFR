"""Train a SpinningUp DDPG RADDTLinear actor and run the closed-loop battery FR control test in Julia.

Run from the experiment directory after the Julia ZeroMQ server is available.
The script saves the trained RADDTLinear parameters as JSON for the closed-loop battery FR control test.
"""

import json
import time
import subprocess
import sys
import argparse
from pathlib import Path
sys.path.append('./src/')
sys.path.append('./src/DT/')

from juliaBatteryGymEnvwithZeroMQ import JuliaBatteryGymEnv
from ddpg.ddpgModify import ddpg
from DTAgentFunc import RADDTLinearActorMLPCritic

def format_run_value(value):
    return f"{value:g}".replace("-", "m").replace(".", "p")

def env_fn():
    max_epochweeks = 1                  # 1 * 24 * 7 = 168 hours, one week
    return JuliaBatteryGymEnv(max_epochweeks=max_epochweeks)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--treeDepth", type=int, default=4)
    parser.add_argument("--scale", type=float, default=25)
    args = parser.parse_args()

    epochsNum = 50
    maxTNum = 168
    hidden_sizes = [30, 15]
    steps_per_epochNum = maxTNum
    total_steps = epochsNum * steps_per_epochNum
    start_stepsNum =  0
    update_afterNum = maxTNum
    update_everyNum = maxTNum
    replay_sizeNum = 10 * maxTNum  # 10 weeks of data
    batch_sizeNum = 160
    gammaNum = 0.9
    polyakNum = 0.99
    act_noiseNum = 0.05
    num_test_episodesNum = 1

    run_id = (
        f"seed{args.seed}"
        f"_D{args.treeDepth}"
        f"_scale{format_run_value(args.scale)}"
    )

    train_log_parent = Path("log") / "Train"
    dt_json_dir = Path("TrainedRADDTLinearJson") / run_id
    control_output_dir = Path("out") / "RADDTLinearDDPGTest" / run_id
    test_log_dir = Path("log") / "BatteryFRControlTest" / run_id
    train_log_parent.mkdir(parents=True, exist_ok=True)
    dt_json_dir.mkdir(parents=True, exist_ok=True)
    control_output_dir.mkdir(parents=True, exist_ok=True)
    test_log_dir.mkdir(parents=True, exist_ok=True)

    from spinup.utils.run_utils import setup_logger_kwargs
    logger_kwargs = setup_logger_kwargs(exp_name=run_id, data_dir='./log/Train/', datestamp=False)



    print("############################### Training DDPG with ZeroMQ ###############################")
    print(f"Run id: {run_id}")
    print(f"Training log dir: {train_log_parent / run_id}")
    print(f"RADDTLinear actor JSON dir: {dt_json_dir}")
    print(f"Battery FR control test output dir: {control_output_dir}")
    print(f"RADDTLinear actor parameters: treeDepth={args.treeDepth}, scale={args.scale:g}")
    print(f"Total steps: {total_steps}, Start steps: {start_stepsNum}, Update after: {update_afterNum}, Update every: {update_everyNum}")



    modelAC = ddpg(env_fn=env_fn,
        actor_critic=RADDTLinearActorMLPCritic,
        ac_kwargs=dict(hidden_sizes=hidden_sizes),
        steps_per_epoch=steps_per_epochNum,
        epochs=epochsNum,
        replay_size=replay_sizeNum,
        seed=args.seed,
        treeDepth=args.treeDepth,
        scale=args.scale,
        batch_size=batch_sizeNum,
        gamma=gammaNum,
        polyak=polyakNum,
        act_noise=act_noiseNum,
        max_ep_len=maxTNum,        
        update_after=update_afterNum,
        update_every=update_everyNum,
        start_steps=start_stepsNum,
        num_test_episodes=num_test_episodesNum,
        logger_kwargs=logger_kwargs
        )



    a_trained = modelAC.pi.linear1.weight.clone().detach()
    b_trained = modelAC.pi.linear1.bias.clone().detach() * (-1.0)
    k_trained = modelAC.pi.k_leafSlope.clone().detach()
    h_trained = modelAC.pi.h_leafIntercept.clone().detach()
    tree_trained = {
        "a": a_trained.tolist(),
        "b": b_trained.tolist(),
        "k": k_trained.tolist(),
        "h": h_trained.tolist(),
        "treeDepth": args.treeDepth,
        "scale": args.scale,
    }
    dt_json_file = dt_json_dir / f"DDPG_RADDTLinearTrained_{run_id}.json"
    with open(dt_json_file, "w") as json_file:
        json.dump(tree_trained, json_file)
    print(f"Saved RADDTLinear actor JSON: {dt_json_file}")

    print("RADDTLinear JSON file saved, waiting 10 seconds before Julia runs...")
    time.sleep(10)
    print("Wait over, now launch Julia.")
    print("############################### Training DDPG with ZeroMQ Completed ###############################")






    print("############################### Closed-loop battery FR control test ###############################")

    with open(test_log_dir / f"stdout_{run_id}.log", "w") as out_file, open(test_log_dir / f"stderr_{run_id}.log", "w") as err_file:
        result = subprocess.run(
            [
                "julia",
                "src/RLTestRunbyPython.jl",
                run_id,
            ],
            stdout=out_file,
            stderr=err_file
        )

    print(f"Battery FR control test return code: {result.returncode}")
    print(f"Battery FR control test logs: {test_log_dir}")
    print("############################### Closed-loop battery FR control test Completed ###############################")
    



