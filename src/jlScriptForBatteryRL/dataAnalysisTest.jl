module dataAnalysisTest
__precompile__() 


using Statistics
using DelimitedFiles
using Distributions
using LinearAlgebra


using assignedValues

export dataProcessTest

function dataProcessTest()

    dataFolderDir = "../data/"
    FR_price = readdlm(dataFolderDir * "FR/FR_Incentive.csv", ',', Float64)
    FR_price_orig = vcat(FR_price'...)
    FR_price = FR_price_orig.+1e-4           
    
    grid_price = readdlm(dataFolderDir * "FR/slow_Price.csv", ',', Float64) 
    grid_price_orig = vcat(grid_price'...)
    
    Files = [dataFolderDir * "FR/01_2017_Dynamic.csv", dataFolderDir * "FR/02_2017_Dynamic.csv", dataFolderDir * "FR/03_2017_Dynamic.csv", dataFolderDir * "FR/04_2017_Dynamic.csv",
                dataFolderDir * "FR/05_2017_Dynamic.csv", dataFolderDir * "FR/06_2017_Dynamic.csv", dataFolderDir * "FR/07_2017_Dynamic.csv", dataFolderDir * "FR/08_2017_Dynamic.csv",
                dataFolderDir * "FR/09_2017_Dynamic.csv", dataFolderDir * "FR/10_2017_Dynamic.csv", dataFolderDir * "FR/11_2017_Dynamic.csv", dataFolderDir * "FR/12_2017_Dynamic.csv"]


    global signal = []

    for i in 1:12
        filename = Files[i]
        originalsignal = readdlm(filename, ',', Float64)[1:(end-1),:]  
        if i == 1
            global signal = vcat(originalsignal...)
        else
            global signal = [signal; vcat(originalsignal...)]
        end
        if i == 12
           global signal = [signal; originalsignal[end,end]]
        end
    end
    
    signal = min.(max.(signal, -1),1)
    signal = signal[1:Nt_FR]
                
                
                
    return  FR_price_orig, grid_price_orig, signal
end

end # module dataAnalysis