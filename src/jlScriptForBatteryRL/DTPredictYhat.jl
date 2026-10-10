module DTPredictYhat
using Statistics

export DTPredYhat_Linear, DTPredYhat_Constant, MSER2Cal


function DTPredYhat_Constant(X::Matrix{Float64}, tree::Dict{String, <:AbstractArray})
    n, p = size(X)
    numTb = length(tree["b"])
    treeDepth = Int(floor(log2(numTb+1)))
    Tb = 2^treeDepth - 1
    t = ones(Int, n)

    for _ in 1:treeDepth
        a_t = tree["a"][t, :]              
        b_t = tree["b"][t]                 

        decisions = sum(a_t .* X, dims=2) .> b_t   
        t = ifelse.(decisions[:], 2 .* t .+ 1, 2 .* t)
    end

    leaf_indices = t .- Tb 
    y_hat = tree["c"][leaf_indices, :]        
    
    return y_hat
end 



function DTPredYhat_Linear(X::Matrix{Float64}, tree::Dict{String, <:AbstractArray})
    n, p = size(X)
    ydim = 2
    numTb = length(tree["b"])
    treeDepth = Int(floor(log2(numTb+1)))
    Tb = 2^treeDepth - 1
    t = ones(Int, n)

    for _ in 1:treeDepth
        a_t = tree["a"][t, :]              
        b_t = tree["b"][t]                 

        decisions = sum(a_t .* X, dims=2) .> b_t  
        t = ifelse.(decisions[:], 2 .* t .+ 1, 2 .* t)
    end


    leaf_indices = t .- Tb 

    k_all = tree["k"]             
    h_all = tree["h"]             

    y_hat = Array{Float32}(undef, n, ydim)

    for i in 1:n
        k = k_all[leaf_indices[i], :, :]     
        h = h_all[leaf_indices[i], :]       
        y_hat[i, :] = k * X[i, :] + h          
    end

    return y_hat
end




function MSER2Cal(Yhat::Matrix, y::Matrix)
    """
    Compute MSE and mean R² for multi-output regression.

    Arguments:
    - Yhat::Matrix{<:Real}: predicted values (n × d)
    - y::Matrix{<:Real}: true values (n × d)

    Returns:
    - mse::Float64: mean squared error across all elements
    - r2_mean::Float64: mean R² score across outputs
    """
    mse = mean((Yhat .- y).^2)  

    mean_y = mean(y, dims=1)                      
    total_var = sum((y .- mean_y).^2, dims=1)     
    residual_var = sum((Yhat .- y).^2, dims=1)    

    r2_per_output = map((tv, rv) -> tv > 1e-8 ? 1 - rv / tv : 0.0,
                        total_var[:], residual_var[:])   
    r2_mean = mean(r2_per_output)

    return mse, r2_mean
end

end 

