module dataAnalysisTrain
__precompile__() 


using Statistics
using DelimitedFiles
using Distributions
using LinearAlgebra

export dataProcessTrain

function dataProcessTrain()


    dataFolderDir = "../data/"
    FR_price = readdlm(dataFolderDir * "FR/FR_Incentive.csv", ',', Float64)
    FR_price_orig = vcat(FR_price'...)
    FR_price = FR_price_orig.+1e-4           # avoid log(0) or log(negative)
    
    grid_price = readdlm(dataFolderDir * "FR/slow_Price.csv", ',', Float64) # size = 365*24
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
                
                
    ### calculate the distribution of grid price
                
    grid_price = grid_price[1:364, :]'
                
                
    Xn_price = reshape(grid_price, (:, 52)) 
    p, n = size(Xn_price)
    Sn = Xn_price * Xn_price' / n
    In = Diagonal(ones(p))
    
    m_n = tr(Sn * In) / p
    d_n2 = norm(Sn - m_n * In)^2
    b_nbar = (1/n^2)*sum(norm(Xn_price[:, i] * Xn_price[:, i]' - Sn)^2 for i = 1:n)
    b_n2 = min(b_nbar, d_n2)
    a_n2 = d_n2 - b_n2
    
    Sigma_n = (b_n2 / d_n2) * m_n * In + (a_n2 / d_n2) * Sn
    mu_n = mean(Xn_price, dims = 2)
    mu_n = dropdims(mu_n, dims = 2)
    d_grid_price = MvNormal(mu_n, Sigma_n)

    
    FR_price_mean = mean(log.(FR_price))
    FR_price_var = std(log.(FR_price))
    
    d_FR_price = LogNormal(FR_price_mean, FR_price_var)
    
    max_grid_price = maximum(vcat(grid_price...))
    min_grid_price = minimum(vcat(grid_price...))
    max_FR_price = maximum(FR_price_orig)
    min_FR_price = minimum(FR_price_orig)

    signal_index = collect(1:1800:(size(signal)[1]-1800*24*28))      
                
    return  d_FR_price, min_FR_price, max_FR_price, d_grid_price,  min_grid_price, max_grid_price, signal, signal_index, FR_price_orig, grid_price_orig
end

end # module dataAnalysis