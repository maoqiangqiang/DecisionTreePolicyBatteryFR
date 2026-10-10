# Decision Tree Policies for Battery Frequency Regulation

This repository corresponds to our study entitled **From black-box neural actors to interpretable decision tree policies: reinforcement learning for battery energy storage in frequency regulation markets**, which is currently under review. All rights are reserved.

`DecisionTreePolicyBatteryFR` learns differentiable decision tree policies for battery frequency regulation (FR) through gradient-based reinforcement learning.

<div align="center">

![Python](https://img.shields.io/badge/language-Python-blue?logo=python)
![Julia](https://img.shields.io/badge/language-Julia-purple?logo=julia)
![PyTorch](https://img.shields.io/badge/dependencies-PyTorch-orange?logo=pytorch)
[![License](https://img.shields.io/badge/license-GPL--3.0-blue)](LICENSE.md)

</div>

## Repository Overview

The repository is organized into **two branches**, which share the battery environment and DDPG training framework:

- `RADDT_Actor`: Tree with constant prediction; each leaf contains a constant action vector.
- `RADDTLinear_Actor`: Tree with linear prediction; each leaf contains an affine model of the observation.

**Current branch: `RADDT_Actor`.** This branch uses **constant predictions at the leaves**.

## Requirements and Installation

Both **Python and Julia** are required:

- **Python** implements the PyTorch decision tree actor and the gradient-based RL training loop.
- **Julia** runs the published software for high-fidelity battery simulation, which serves as the RL environment.

**ZeroMQ** and **JuliaCall** enable communication between Python and Julia.

### Python Environment

Install the following Python packages:

- `torch`
- `numpy`
- `scipy`
- `h5py`
- `gym`
- `pyzmq`
- `juliacall`
- `spinup`

### Julia Environment

Install Julia and make sure the `julia` command is available on `PATH`.

Required Julia packages include:

- `DifferentialEquations`, `Sundials`, `Interpolations`, `Distributions`
- `JuMP`, `Ipopt`, `GLPK`
- `Statistics`, `DelimitedFiles`, `LinearAlgebra`, `StatsBase`, `Random`
- `JLD`, `BSON`, `ZMQ`, `JSON`, `CSV`, `DataFrames`

From the root of the checked-out branch, install the Julia dependencies using:

```shell
python test/install_juliaUsedPackage.py
```

Use the same Julia environment for JuliaCall and the standalone `julia` command.

## Script Description

### `test/`: Experiment Entry Points

- `test_RADDT_DDPG_RL_TrainandTest.py`: Complete experiment script for DDPG reinforcement learning training and Julia-based closed-loop battery FR control testing of the trained decision tree policy.
- `install_juliaUsedPackage.py`: Installs the Julia dependencies through JuliaCall.

### `src/ddpg/`: Reinforcement Learning

The code is adapted from the standard PyTorch DDPG implementation in SpinningUp, with modifications for the decision tree actor and battery environment.

DDPG (Deep Deterministic Policy Gradient) is the example RL algorithm in this repository. **The differentiable decision tree policy is compatible with standard gradient-based RL and is not restricted to DDPG.** Integration with other algorithms requires adapting their policy interface and training setup.

- `core.py`: Neural network building blocks for the actor and critic.
- `ddpgModify.py`: Modified DDPG training loop, including experience replay, actor/critic updates, logging, and evaluation.

### `src/DT/`: Differentiable Decision Tree Policy

RADDT is a **"ReLU+Argmin"-based Differentiable Decision Tree**, an original differentiable decision tree optimization method. Algorithm details are available in the [original RADDT repository](https://github.com/maoqiangqiang/RADDT).

- `DTAgentFunc.py`: Defines the differentiable decision tree actor with constant leaf predictions and its actor–critic interface.

### `src/`: Python–Julia Interface and Evaluation

- `juliaBatteryGymEnvwithZeroMQ.py`: Python wrapper for interacting with the Julia battery environment through ZeroMQ and JuliaCall.
- `juliaBatteryServer.jl`: Julia server that receives actions from Python and returns battery observations and rewards.
- `RLTestRunbyPython.jl`: Loads the trained decision tree policy and evaluates its performance in Julia-based closed-loop battery FR control.

### `src/jlScriptForBatteryRL/`: Battery Environment

The battery environment is adapted from published Julia software; see the references in our paper for details.

This folder contains Julia scripts for the battery environment that are used during Python–Julia communication and policy evaluation.

### `sh_slurm/`: Cluster Submission

- `demoSlurm.sh`: Example Slurm job script that starts the Julia server and runs DDPG reinforcement learning training followed by closed-loop battery FR control testing. Update the paths and environment settings for your cluster.

## Usage Example

Run the following commands from the **root of the checked-out branch**, where `src/` and `test/` are located.

Start the Julia battery server in one terminal:

```shell
julia src/juliaBatteryServer.jl
```

Once the server has initialized, run the complete experiment in a second terminal with the Python environment activated:

```shell
python test/test_RADDT_DDPG_RL_TrainandTest.py --seed 42 --treeDepth 3 --scale 20
```

The script automatically launches Julia-based closed-loop battery FR control testing after DDPG reinforcement learning training. Other training settings can be changed inside the experiment script.

The server and client use port `5555` by default. To use another port, set `BATTERY_ZMQ_PORT` to the same value in both terminals.

For Slurm, submit the example job using:

```shell
sbatch sh_slurm/demoSlurm.sh
```

## Others

For questions or unexpected behavior, please contact us at maoq@student.ubc.ca.

## License

This repository is released under the [GNU General Public License v3.0](LICENSE.md).
