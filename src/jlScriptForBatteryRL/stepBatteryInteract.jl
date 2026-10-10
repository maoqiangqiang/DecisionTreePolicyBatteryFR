module stepBatteryInteract
__precompile__() 


using Statistics
using assignedValues, batterySimSetup


export step_battery, set_reset, set!, reset!, step!








function denormalizeU(u,ind=0)
    if ind==0
        return (umax .+ umin)/2 .+ (umax .- umin) ./ 2 .* u
    else
        return (umax[ind] .+ umin[ind])/2 .+ (umax[ind] .- umin[ind]) ./ 2 .* u[ind]
    end
end


function step!(env, action::Array{A,1}, FR_price_segment, grid_price_segment, signal_segment) where {S,A,R}
    u0ff = env.state                         
    ctr = denormalizeU(action)
    FR_band = ctr[1]
    grid_band = ctr[2]
   

    soc_end, capacity_remain, u0ff, reward1, reward2, done = step_battery(FR_price_segment, grid_price_segment, signal_segment, FR_band, grid_band, u0ff)
    env.state = u0ff
    env.current[5:6] .= soc_end, capacity_remain
    return env.current, reward1, reward2, done
end

function reset!(env)
    env.state = deepcopy(assignedValues.u_start)
    env.current = [0,0,0,0,0.5,1] 
end


function set!(env, FR_price, grid_price, signal)
    env.current[1:4] .= Statistics.mean(signal), Statistics.var(signal), FR_price[1], grid_price[1]
end







function step_battery(FR_price_segment, grid_price_segment, signal_segment, FR_band, grid_band, u0ff)

    cf0 = u0ff[Ncp+Ncn+Nsei+8]
    fade = cf0/Qmax
    capacity_remain = 1 - fade
    csn_avg = u0ff[Ncp+1]
    soc = csn_avg/csnmax

    soc_stop = true
    TIME_FR_segment = 0:dt_FR:3600
    soc_end = copy(soc)
    while soc_stop
        soc_end, state_end, simulation_status = simulation(TIME_FR_segment, signal_segment, FR_band, grid_band, u0ff)
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
           
            u0ff = state_end  
            soc_stop = false
        end
    end

    cf_e = u0ff[Ncp+Ncn+Nsei+8]
    fade = cf_e/Qmax
    capacity_remain = 1 - fade

    done = (capacity_remain <= capacity_retire)
    expectedrevenue = 241407*P_nominal/1000
    reward1 = sum(FR_price_segment[h] * FR_band[h] - max(0,grid_band[h])*grid_price_segment[h] for h in 1:nHours_Horizon) 
    reward2 = (sum(FR_price_segment[h] * FR_band[h] - max(0,grid_band[h])*grid_price_segment[h] for h in 1:nHours_Horizon)- 5*(soc_end - 0.5*capacity_remain)^2 -expectedrevenue/(1-capacity_retire)/Qmax * (cf_e - cf0))


    return soc_end, capacity_remain, u0ff, reward1, reward2, done
end



function set_reset(FR_band, grid_band, signal_segment, u0f)
    P_FR_segment = signal_segment*FR_band .+ grid_band
    csp_avg0 = u0f[1]
    csn_avg0 = u0f[Ncp+1]
    delta_sei0 = u0f[Ncp+Ncn+7]
    csp_s0, csn_s0, iint0, phi_p0, phi_n0, pot0, it0, isei0 = getinitial(csp_avg0, csn_avg0, delta_sei0,  P_FR_segment[1], 3)
    u0f[Ncp] = csp_s0
    u0f[Ncp+Ncn] = csn_s0
    u0f[Ncp+Ncn+1] = iint0                  #iint
    u0f[Ncp+Ncn+2] = phi_p0                 #phi_p
    u0f[Ncp+Ncn+3] = phi_n0                 #phi_n
    u0f[Ncp+Ncn+4] = pot0                   #pot
    u0f[Ncp+Ncn+5] = it0                     #it
    u0f[Ncp+Ncn+6] = isei0                   #isei

    return u0f

end 


end 

