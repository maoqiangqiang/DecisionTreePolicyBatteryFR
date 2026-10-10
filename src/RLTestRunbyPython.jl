using JSON

if !("src/" in LOAD_PATH)
    push!(LOAD_PATH, "src/")
    push!(LOAD_PATH, "src/jlScriptForBatteryRL/")
end

using dataAnalysisTest
using MPCControl_RL
using assignedValues



##################################### Load Trained DT Agent ######################################
println("################# Load Trained DT Agent ##################")

runId = length(ARGS) >= 1 ? ARGS[1] : "seed42_D4_scale25"
jsonFileAddr = joinpath("TrainedRADDTJson", runId, string("DDPG_RADDTTrained_", runId, ".json"))
controlOutputDir = joinpath("out", "RADDTDDPGTest", runId)

println("Run id: ", runId)
println("DT JSON file: ", jsonFileAddr)
println("Battery FR control test output dir: ", controlOutputDir)

file = open(jsonFileAddr, "r")
treeJsonDict = JSON.parse(file)
close(file)

tree_a = hcat(treeJsonDict["a"]...)' |> Array{Float32, 2}
tree_b = Float32.(treeJsonDict["b"])
tree_c = hcat(treeJsonDict["c"]...)' |> Array{Float32, 2}

treeDict = Dict(
    "a" => tree_a,
    "b" => tree_b,
    "c" => tree_c,
)

println("DT actor loaded successfully!")





# ################################### Closed-loop battery FR control test with the trained RL policy ######################################
println("################# Closed-loop battery FR control test with the trained RL policy ##################")

FR_price_orig, grid_price_orig, signal = dataProcessTest()


batch = 1
trainedModelRLCase = "noInit"
u_start_used = deepcopy(assignedValues.u_start)
MPC_RL(u_start_used, batch, treeDict, trainedModelRLCase, FR_price_orig, grid_price_orig, signal, controlOutputDir, runId)
