using ZMQ
using JSON

push!(LOAD_PATH, "./src/jlScriptForBatteryRL/")
using RLEnvironments
using assignedValues
using stepBatteryInteract

ctx = Context()
socket = Socket(ctx, REP)
zmq_port = get(ENV, "BATTERY_ZMQ_PORT", "5555")
zmq_endpoint = "tcp://127.0.0.1:$(zmq_port)"
bind(socket, zmq_endpoint)

println("Julia ZeroMQ server running at $(zmq_endpoint)")

env = RLEnvironments.FR_battery{Float64}(assignedValues.state_dim, assignedValues.inner_state_dim)
stepBatteryInteract.reset!(env)

env_backup = nothing

function set_current_from_input!(env, input)
    FR_price = Float64.(input["FR_price"])
    grid_price = Float64.(input["grid_price"])
    signal = Float64.(input["signal"])
    t_Episode = input["t_Episode"]

    fr = FR_price[t_Episode + 1]
    grid = grid_price[t_Episode + 1]
    signal_seg = signal[(t_Episode * assignedValues.Nt_FR_hour + 1):((t_Episode + 1) * assignedValues.Nt_FR_hour + 1)]
    stepBatteryInteract.set!(env, fr, grid, signal_seg)

    return fr, grid, signal_seg
end

while true
    msg = String(recv(socket))
    input = JSON.parse(msg)

    if haskey(input, "reset") && input["reset"] == true
        stepBatteryInteract.reset!(env)
        set_current_from_input!(env, input)
        send(socket, JSON.json(Dict("obs" => copy(env.current))))
        println("Julia server: Environment reset complete. Sending confirmation to Python.")
        continue
    end

    if haskey(input, "set_current") && input["set_current"] == true
        set_current_from_input!(env, input)
        send(socket, JSON.json(Dict("obs" => copy(env.current))))
        continue
    end

    if haskey(input, "save_state") && input["save_state"] == true
        global env_backup = deepcopy(env)
        send(socket, "state_saved")
        println("Julia server: Environment state saved.")
        continue
    end

    if haskey(input, "restore_state") && input["restore_state"] == true
        if env_backup === nothing
            send(socket, "no_backup")
            println("Julia server: No backup to restore!")
        else
            global env = deepcopy(env_backup)
            send(socket, "state_restored")
            println("Julia server: Environment state restored.")
        end
        continue
    end


    action = Float64.(input["action"])
    FR_price = Float64.(input["FR_price"])
    t_Episode = input["t_Episode"]

    fr, grid, signal_seg = set_current_from_input!(env, input)
    obsPrime, reward1, reward2, done_flag = stepBatteryInteract.step!(env, action, fr, grid, signal_seg)
    next_t = t_Episode + 1
    epoch_done = next_t >= length(FR_price)

    if !done_flag && !epoch_done
        input_next = copy(input)
        input_next["t_Episode"] = next_t
        set_current_from_input!(env, input_next)
        obs = copy(env.current)
    else
        obs = copy(obsPrime)
    end

    println("obs = ", obs)

    result = Dict(
        "obs" => obs,
        "reward1" => reward1,
        "reward2" => reward2,
        "done" => done_flag,
        "epoch_done" => epoch_done
    )

    send(socket, JSON.json(result))
end
