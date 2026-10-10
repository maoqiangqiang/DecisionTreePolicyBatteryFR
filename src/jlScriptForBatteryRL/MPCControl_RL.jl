module MPCControl_RL
__precompile__() 

using JuMP
using GLPK
using JLD 
using Statistics
using CSV
using DataFrames

using batterySimSetup, assignedValues, DTPredictYhat

export MPC_RL




function MPC_RL(u0f, batch, actor, trainedModelRLCase, FR_price_sample, grid_price_sample, signal, outputRunDir="", runId="")
    
    println("Closed-loop battery FR control test started")
    baseSaveFolder = isempty(outputRunDir) ? joinpath("out", "RADDTLinearDDPGTest") : outputRunDir
    fileNameSuffix = isempty(runId) ? "" : string("_", runId)
    saveFolder = ""
    fileNamePrefix = ""

    OnlinePredTime = Float64[]

    start_time = time()



    FR_band_list = zeros(TotalHours)
    buy_from_grid = zeros(TotalHours)
    capacity_remain_list = ones(TotalHours)
    soc_list = ones(TotalHours)
    
    retire_time = Totaltime
    retire_index = Nt_FR

    X_TRAIN = []
    Y_TRAIN = []

    for i_start in 1:Nt_FR_hour:(Nt_FR-Nt_FR_hour)
        i_end = i_start + Nt_FR_hour
        i_start_hour = Int(floor(TIME_FR[i_start]/3600))+1
        cf0 = u0f[Ncp+Ncn+Nsei+8]
        fade = cf0/Qmax
        capacity_remain = 1 - fade
        csn_avg = u0f[Ncp+1]
        soc = csn_avg/csnmax
        capacity_remain_list[i_start_hour] = capacity_remain
        soc_list[i_start_hour] = soc
        if capacity_remain <= capacity_retire || i_start >= Nt_FR-Nt_FR_hour
            retire_time = TIME_FR[i_start]
            retire_index = i_start


            if trainedModelRLCase == "Init"
                saveFolder = joinpath(baseSaveFolder, "Init")
                isdir(saveFolder) || mkpath(saveFolder)
                fileNamePrefix = string("RADDTLinearDDPGTestInit", fileNameSuffix)
                filename_Sim = joinpath(saveFolder, string(fileNamePrefix,"_SimuBatch", batch, ".jld"))
                save(filename_Sim, "retire_time",retire_time,  "FR_band_list",FR_band_list, "buy_from_grid",buy_from_grid,
                "capacity_remain_list", capacity_remain_list, "soc_list", soc_list)
            elseif trainedModelRLCase == "noInit"
                saveFolder = joinpath(baseSaveFolder, "noInit")
                isdir(saveFolder) || mkpath(saveFolder)
                fileNamePrefix = string("RADDTLinearDDPGTestnoInit", fileNameSuffix)
                filename_Sim = joinpath(saveFolder, string(fileNamePrefix,"_SimuBatch", batch, ".jld"))
                save(filename_Sim, "retire_time",retire_time,  "FR_band_list",FR_band_list, "buy_from_grid",buy_from_grid,
                "capacity_remain_list", capacity_remain_list, "soc_list", soc_list)
            else
                error("Unsupported trainedModelRLCase: $(trainedModelRLCase)")
            end

            break
        end


        signal_segment = signal[i_start:i_end]

        FR_price_segment = FR_price_sample[i_start_hour:i_start_hour]
        grid_price_segment = grid_price_sample[i_start_hour:i_start_hour]


        onlinePredTimeStart = time()
        FR_band, grid_band = OptimalControl(u0f, actor,  signal_segment,  FR_price_segment, grid_price_segment)
        onlinePredTimeEnd = time()
        push!(OnlinePredTime, onlinePredTimeEnd - onlinePredTimeStart)


        FR_band = max(min(FR_band,umax[1]),umin[1])
        grid_band = max(min(grid_band,umax[2]),umin[2])


        #### Simulate with the optimal FR_band, grid_band
        if TIME_FR[i_start]%(3600*24) == 0
            println("TIME_FR_segment (day)  ", TIME_FR[i_start]/3600/24)
            println("    capacity_remain    ", capacity_remain, "    soc     ", soc)
        end

        soc_stop = true
        while soc_stop
            TIME_FR_segment = TIME_FR[i_start:i_end]
            soc_end, state_end, simulation_status = simulation(TIME_FR_segment, signal_segment, FR_band, grid_band, u0f)
            # if stop prematurely
            if !simulation_status  || soc_end <= 0.1  || soc_end >= 0.9
                if FR_band >= P_nominal
                    FR_band = FR_band - P_nominal/2
                else
                    FR_band = FR_band/2
                end

                power_next = P_nominal*soc + FR_band*mean(signal_segment)
                if power_next <= P_nominal*capacity_remain*soc_min
                    grid_band = P_nominal*capacity_remain*soc_min - power_next
                elseif power_next >= P_nominal*capacity_remain*soc_max
                    grid_band = P_nominal*capacity_remain*soc_max -  power_next
                else
                    grid_band = 0
                end
            else
                FR_band_list[i_start_hour] = FR_band
                buy_from_grid[i_start_hour] = grid_band
                u0f = state_end   # only the simulation process sucsess, then update the initial condition u0f
                soc_stop = false
            end
        end

        train_x = [Statistics.mean(signal_segment), Statistics.var(signal_segment), FR_price_segment[1], grid_price_segment[1],soc,capacity_remain]
        train_y = [FR_band, grid_band]
        if i_start == 1
            X_TRAIN = train_x
            Y_TRAIN = train_y
        else
            X_TRAIN = hcat(X_TRAIN,train_x)
            Y_TRAIN = hcat(Y_TRAIN,train_y)
        end

        cf = u0f[Ncp+Ncn+Nsei+8]
        fade = cf/Qmax
        capacity_remain = 1 - fade
        csn_avg = u0f[Ncp+1]
        soc = csn_avg/csnmax
    end

    println("############################")
    println("batch: ", batch)
    println("size of X_TRAIN: ", size(X_TRAIN))
    println("size of Y_TRAIN: ", size(Y_TRAIN))

    println("elapsed time:     ", time() - start_time)

    isdir(saveFolder) || mkpath(saveFolder)

    filename_TrainXY = joinpath(saveFolder, string(fileNamePrefix, "_TrainingDataBatch",batch,".jld"))
    save(filename_TrainXY, "X", X_TRAIN, "Y", Y_TRAIN)



    println("\nTotal OnlinePred Loop Number: ", length(OnlinePredTime))
    println("Average OnlinePred Time: ", mean(OnlinePredTime))
    df = DataFrame(OnlinePredTime = OnlinePredTime)
    onlinePredTimeFileAddr_csv = joinpath(saveFolder, string(fileNamePrefix, "_OnlinePredTime_Batch", batch, ".csv"))
    CSV.write(onlinePredTimeFileAddr_csv, df)
    # remove the maximum and minimum time and then calculate the average time
    sort!(OnlinePredTime)
    println("Maximum OnlinePred Time: ", maximum(OnlinePredTime))
    println("Minimum OnlinePred Time: ", minimum(OnlinePredTime))
    println("Average OnlinePred Time without max and min: ", mean(OnlinePredTime[2:end-1]))



    Ncell = 1e6/P_nominal
    revenue = 0
    cost = 0


    for i in 1:size(FR_price_sample)[1]
        price = FR_price_sample[i]
        FR_band = FR_band_list[i]
    	revenue += price * FR_band *Ncell/1000
    end

    for i in 1:size(grid_price_sample)[1]
        price = grid_price_sample[i]
        grid_band = buy_from_grid[i]
        if grid_band > 0
            cost += price*grid_band*Ncell/1000
        end
    end



    profit = revenue - cost
    println("\nretire_time:  ", retire_time)
    println("revenue,   operational cost,  profit:    ", revenue, "   ", cost, "   ",profit)
    println("charging times(h):  ", length(findall(buy_from_grid.>0)))
    println("discharging times(h):  ", length(findall(buy_from_grid.<0)))
    
end



function Denormalize_input(u)
    return (umax .+ umin)/2 .+ (umax .- umin)/2 .* u
end


function OptimalControl(u0ff, treeModel, signal_segment, FR_price_segment, grid_price_segment)

    cf0 = u0ff[Ncp+Ncn+Nsei+8]
    fade = cf0/Qmax
    capacity_remain = 1 - fade

    csp_avg0 = u0ff[1]
    csn_avg0 = u0ff[Ncp+1]
    soc = csn_avg0/csnmax
    x = [Statistics.mean(signal_segment), Statistics.var(signal_segment), FR_price_segment[1], grid_price_segment[1],soc,capacity_remain]
    X_matrix = reshape(x, 1, :)
    yhat = tanh.(DTPredYhat_Linear(X_matrix, treeModel))
    ctr = Denormalize_input(yhat)
    return ctr[1], ctr[2]

end














end 







