using JSON

if !("src/" in LOAD_PATH)
    push!(LOAD_PATH, "src/")
    push!(LOAD_PATH, "src/jlScriptForBatteryRL/")
end

using dataAnalysisTest
using MPCControl_RL
using assignedValues



##################################### Load Trained RADDTLinear Agent ######################################
println("################# Load Trained RADDTLinear Agent ##################")

runId = length(ARGS) >= 1 ? ARGS[1] : "seed42_D4_scale25"
jsonFileAddr = joinpath("TrainedRADDTLinearJson", runId, string("DDPG_RADDTLinearTrained_", runId, ".json"))
controlOutputDir = joinpath("out", "RADDTLinearDDPGTest", runId)

println("Run id: ", runId)
println("RADDTLinear JSON file: ", jsonFileAddr)
println("Battery FR control test output dir: ", controlOutputDir)

file = open(jsonFileAddr, "r")
treeJsonDict = JSON.parse(file)
close(file)

tree_a = hcat(treeJsonDict["a"]...)' |> Array{Float32, 2}
tree_b = Float32.(treeJsonDict["b"])
tree_h = hcat(treeJsonDict["h"]...)' |> Array{Float32, 2}

tree_k_raw = treeJsonDict["k"]
numLeaves = length(tree_k_raw)
ydim = length(tree_k_raw[1])
p = length(tree_k_raw[1][1])
tree_k = Array{Float32}(undef, numLeaves, ydim, p)

for leafIdx in 1:numLeaves
    for yIdx in 1:ydim
        for xIdx in 1:p
            tree_k[leafIdx, yIdx, xIdx] = Float32(tree_k_raw[leafIdx][yIdx][xIdx])
        end
    end
end

treeDict = Dict(
    "a" => tree_a,
    "b" => tree_b,
    "k" => tree_k,
    "h" => tree_h,
)

println("RADDTLinear actor loaded successfully!")





# ################################### Closed-loop battery FR control test with the trained RL policy ######################################
println("################# Closed-loop battery FR control test with the trained RL policy ##################")

FR_price_orig, grid_price_orig, signal = dataProcessTest()


batch = 1
trainedModelRLCase = "noInit"
u_start_used = deepcopy(assignedValues.u_start)
MPC_RL(u_start_used, batch, treeDict, trainedModelRLCase, FR_price_orig, grid_price_orig, signal, controlOutputDir, runId)
