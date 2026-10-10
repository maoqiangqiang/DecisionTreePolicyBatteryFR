import gym
import numpy as np
from gym import spaces
import zmq
import json
import os
from juliacall import Main as jl


# === Global ZeroMQ context/socket ===
_zmq_context = zmq.Context()
_zmq_socket = _zmq_context.socket(zmq.REQ)
_zmq_port = os.environ.get("BATTERY_ZMQ_PORT", "5555")
_zmq_endpoint = f"tcp://127.0.0.1:{_zmq_port}"
_zmq_socket.connect(_zmq_endpoint)
print(f"Python ZeroMQ client connected to {_zmq_endpoint}")


# === Load Julia scripts (only once) ===
jl.seval('push!(LOAD_PATH, "./src/jlScriptForBatteryRL/")')
jl.include("./src/jlScriptForBatteryRL/assignedValues.jl")
jl.include("./src/jlScriptForBatteryRL/dataAnalysisTrain.jl")
jl.include("./src/jlScriptForBatteryRL/dataAnalysisTest.jl")
jl.seval("using Random; Random.seed!(970425)")
jl.seval("""
global d_FR_price = nothing
global d_grid_price = nothing
global epoch_week = nothing
global min_FR_price = nothing
global max_FR_price = nothing
global signal = nothing
global signal_index = nothing
global Nt_FR_hour = nothing
""")

(d_FR_price, min_FR_price, max_FR_price,
d_grid_price, min_grid_price, max_grid_price,
signal, signal_index, FR_price_orig, grid_price_orig) = jl.dataAnalysisTrain.dataProcessTrain()

Fixed_FR_price_orig, Fixed_grid_price_orig, Fixed_signal = jl.dataAnalysisTest.dataProcessTest()

Nt_FR_hour = int(jl.assignedValues.Nt_FR_hour)

class JuliaBatteryGymEnv(gym.Env):
    def __init__(self, max_epochweeks):
        super(JuliaBatteryGymEnv, self).__init__()
        self.state_dim = 6
        self.action_dim = 2
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.state_dim,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1, high=1, shape=(self.action_dim,), dtype=np.float32)
        self.max_epochweeks = max_epochweeks
        self.t_Episode = 0

        # Copy Julia arrays
        self.d_FR_price = d_FR_price
        self.d_grid_price = d_grid_price
        self.signal = np.array(signal)
        self.signal_index = np.array(signal_index)

        self.testing_mode = False
        self.env_ready = False



    def prepareEpochData(self):
        jl.d_FR_price = d_FR_price
        jl.d_grid_price = d_grid_price
        jl.epoch_week = self.max_epochweeks
        jl.min_FR_price = min_FR_price
        jl.max_FR_price = max_FR_price
        jl.signal = signal
        jl.signal_index = signal_index
        jl.Nt_FR_hour = Nt_FR_hour




        fr_sample, grid_sample, signal_segment_smaple = jl.seval(f"""
        let
            fr = min.(max.(rand(d_FR_price, 7*epoch_week*24), min_FR_price), max_FR_price)

            grid = (1/50) * vcat(sum(rand(d_grid_price, epoch_week) for _ in 1:50)...)

            signal_start = rand(signal_index)
            signal_end   = signal_start + Nt_FR_hour * (epoch_week * 24 * 7) + 1
            sig = signal[signal_start:signal_end]

            (fr, grid, sig)
        end
        """)

    

        self.FR_price        =  np.array(fr_sample)
        self.grid_price      =  np.array(grid_sample)
        self.signal_segment  =  np.array(signal_segment_smaple)


        self.t_Episode = 0                 
        if self.env_ready:
            return self.set_current_obs()
        return None






    def _current_request_data(self):
        return {
            "FR_price": self.FR_price.tolist(),
            "grid_price": self.grid_price.tolist(),
            "signal": self.signal_segment.tolist(),
            "t_Episode": self.t_Episode
        }


    def set_current_obs(self):
        input_data = {"set_current": True}
        input_data.update(self._current_request_data())
        _zmq_socket.send_string(json.dumps(input_data))
        result = json.loads(_zmq_socket.recv())
        return np.array(result["obs"], dtype=np.float32)



    def reset(self):
        self.t_Episode = 0

        if self.testing_mode:
            self.FR_price = np.array(Fixed_FR_price_orig)[0: self.max_epochweeks * 24 * 7]
            self.grid_price = np.array(Fixed_grid_price_orig)[0: self.max_epochweeks * 24 * 7]
            self.signal_segment = np.array(Fixed_signal)[0: Nt_FR_hour * (self.max_epochweeks * 24 * 7) + 1]

        else:
            assert hasattr(self, "FR_price") and hasattr(self, "grid_price"), \
                "Call prepare_epoch_data() before reset!"


        # Reset the battery state in the Julia server.
        input_data = {"reset": True}
        input_data.update(self._current_request_data())
        _zmq_socket.send_string(json.dumps(input_data))
        result = json.loads(_zmq_socket.recv())
        obs = np.array(result["obs"], dtype=np.float32)
        self.env_ready = True
        return obs
    

    def save_env_state(self):
        _zmq_socket.send_string(json.dumps({"save_state": True}))
        ackSaveState = _zmq_socket.recv()
        print("ackSaveState from Julia after save_env_state:", ackSaveState)

    def restore_env_state(self):
        _zmq_socket.send_string(json.dumps({"restore_state": True}))
        ackRestoreState = _zmq_socket.recv()
        print("ackRestoreState from Julia after restore_env_state:", ackRestoreState)


    def step(self, action):
        input_data = {
            "action": action.tolist(),
            "FR_price": self.FR_price.tolist(),
            "grid_price": self.grid_price.tolist(),
            "signal": self.signal_segment.tolist(),
            "t_Episode": self.t_Episode
        }

        _zmq_socket.send_string(json.dumps(input_data))
        result = json.loads(_zmq_socket.recv())

        obs = np.array(result["obs"], dtype=np.float32)
        reward = result["reward2"]
        done_flag = result["done"]



        self.t_Episode += 1
        done = done_flag
        info = {"reward1": result["reward1"], "reward2": result["reward2"], "epoch_done": result["epoch_done"]}
        return obs, reward, done, info
