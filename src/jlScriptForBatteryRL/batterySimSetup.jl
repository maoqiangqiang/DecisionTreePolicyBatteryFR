module batterySimSetup
__precompile__() 

using DifferentialEquations
using Statistics
using Sundials
using Interpolations
using Distributions
using JuMP
using Ipopt

using assignedValues

export simulation






function simulation(TIME_FR_segment, signal_segment, FR_band, grid_band, u0f)

	TIME_FR_segment = 0:dt_FR:3600
	P_FR_segment = signal_segment*FR_band .+ grid_band
    cf0 = u0f[Ncp+Ncn+Nsei+8]
    fade = cf0/Qmax
    capacity_remain = 1 - fade
	du0 = zeros(Ncp+Ncn+4+Nsei+Ncum)
	tspan = (float(TIME_FR_segment[1]), float(TIME_FR_segment[end]))
    itp = interpolate((TIME_FR_segment,), P_FR_segment, Gridded(Linear()))
    
    function stop_cond(u,t,integrator)
        csn_avg = u[Ncp+1]
        soc_in = csn_avg/csnmax
        min_stop = capacity_remain*soc_min_stop
        max_stop = capacity_remain*soc_max_stop

        if t<=50
            return (min_stop + max_stop)/2
        end

        if soc_in >= (min_stop + max_stop)/2
            return (max_stop - soc_in)
        else
            return (soc_in - min_stop)
        end
    end



    function f_FR(out,du,u,param,t)
        p_to_battery = itp[t]
        it = u[Ncp+Ncn+5]
        phi_p = u[Ncp+Ncn+2]
        phi_n = u[Ncp+Ncn+3]
        p = [p_to_battery]
        f_common(out,du,u,p,t)
        out[end] = (phi_p - phi_n)*it - p_to_battery  
    end

    ### modify algerbric variables
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
    prob = DAEProblem(f_FR, du0, u0f, tspan, differential_vars=differential_vars)
    sol = DifferentialEquations.solve(prob, IDA())
    csn_avg = (sol[end])[Ncp+1]
    soc_end = csn_avg/csnmax
    state_end = sol[end]
	simulation_status = true
	if sol.t[end] != tspan[end]
        simulation_status = false
    end

	return soc_end, state_end, simulation_status  # solve the DAE problem, and return the soc_end = csn_avg/csnmax
end






function f_common(out,du,u,p,t)

    value = p[1]

    csp = u[1:Ncp]
    csn = u[Ncp+1:Ncp+Ncn]
    csp = max.(1, min.(csp, cspmax-1))
    csn = max.(1, min.(csn, csnmax-1))
    csp_avg = csp[1]
    csp_s = csp[2]
    csn_avg = csn[1]
    csn_s = csn[2]

    iint = u[Ncp+Ncn+1]
    phi_p = u[Ncp+Ncn+2]
    phi_n = u[Ncp+Ncn+3]
    pot = u[Ncp+Ncn+4]

    it = u[Ncp+Ncn+5]
    isei = u[Ncp+Ncn+6]
    delta_sei = u[Ncp+Ncn+7]


    cm = u[Ncp+Ncn+5+Nsei]
    cp = u[Ncp+Ncn+6+Nsei]
    Q = u[Ncp+Ncn+7+Nsei]
    cf = u[Ncp+Ncn+8+Nsei]


    #C2. Additional Equaions
    #Positive electrode
    theta_p = csp[Ncp]/cspmax
    Up = 7.49983 - 13.7758*theta_p.^0.5 + 21.7683*theta_p - 12.6985*theta_p.^1.5 + 0.0174967./theta_p -0.41649*theta_p.^(-0.5) -0.0161404*exp.(100*theta_p-97.1069) +
    	0.363031 * tanh.(5.89493 *theta_p -4.21921)
    jp = 2*kp*ce^(0.5)*(cspmax-csp[Ncp])^(0.5)*csp[Ncp]^(0.5)*sinh(0.5*F/R/T*(phi_p-Up))
    out[Ncp+Ncn+1] = (jp - (it/ap/F/lp))

    #Negative electrode
    theta_n = csn[Ncn]/csnmax
    Un = 9.99877 -9.99961*theta_n.^0.5 -9.98836*theta_n + 8.2024*theta_n.^1.5  + 0.23584./theta_n -2.03569*theta_n.^(-0.5) -
         1.47266*exp.(-1.14872*theta_n+2.13185)-
         9.9989*tanh.(0.60345*theta_n-1.58171)
    jn = 2*kn*ce^(0.5)*(csnmax-csn[Ncn])^(0.5)*csn[Ncn]^(0.5)*sinh(0.5*F/R/T*(phi_n-Un+(Rsei+delta_sei/Kappa_sei)*it/an/lnn));
    out[Ncp+Ncn+2] = (jn + (iint/an/F/lnn));
    out[Ncp+Ncn+3] = pot-phi_p + phi_n;

    #C1. Governing Equations
    #Positive electrode
    out[1] = - 3*jp/Rpp - du[1]
    out[2] = 5*(csp_s - csp_avg) + Rpp*jp/Dp

    #Negative electrode
    out[Ncp+1] = - 3 * jn/Rpn - du[Ncp+1]
    out[Ncp+2] = 5*(csn_s - csn_avg) + Rpn*jn/Dn


    #C3. SEI layer Equations
    out[Ncp+Ncn+4] = -iint + it - isei
    out[Ncp+Ncn+5] = -isei+an*lnn*ksei*exp(-1*F/R/T*(phi_n - Urefs + it/an/lnn*(delta_sei/Kappa_sei+Rsei)))
    out[Ncp+Ncn+6] = isei*M_sei/F/rho_sei/an/lnn - du[Ncp+Ncn+7]   #d delta_sei/dt


    #C4. Charge stored
    out[Ncp+Ncn+4 + Nsei] = iint/3600 - du[Ncp+Ncn+5+Nsei];       #dcm/dt
    out[Ncp+Ncn+5 + Nsei] = it*pot/3600  - du[Ncp+Ncn+6+Nsei];       #dcp/dt
    out[Ncp+Ncn+6 + Nsei] = it/3600  - du[Ncp+Ncn+7+Nsei];         #dQ/dt
    out[Ncp+Ncn+7 + Nsei] = isei/3600  - du[Ncp+Ncn+8+Nsei];       #dcf/dt  isei = Cr
end



function  getinitial(csp_avg, csn_avg, delta_sei, value, mode)

    m = Model(optimizer_with_attributes(Ipopt.Optimizer, "print_level" => 0))

    theta_p_guess = min(0.9, csp_avg/cspmax)
    theta_n_guess = min(0.9, csn_avg/csnmax)
    Un_guess = 9.99877 -9.99961*theta_n_guess.^0.5 -9.98836*theta_n_guess + 8.2024*theta_n_guess.^1.5  + 0.23584./theta_n_guess -2.03569*theta_n_guess.^(-0.5) -
         1.47266*exp.(-1.14872*theta_n_guess+2.13185)-
         9.9989*tanh.(0.60345*theta_n_guess-1.58171)
    Up_guess = 7.49983 - 13.7758*theta_p_guess.^0.5 + 21.7683*theta_p_guess - 12.6985*theta_p_guess.^1.5 + 0.0174967./theta_p_guess -0.41649*theta_p_guess.^(-0.5) -
	0.0161404*exp.(100*theta_p_guess-97.1069) +
        0.363031 * tanh.(5.89493 *theta_p_guess -4.21921)

    if mode == 1
        it0 = value
    elseif mode == 2
        it0 = TC/2
    elseif mode == 3
	    pot0 = max(min(Up_guess-Un_guess, 3.3), 2.0)
        it0 = value/pot0
    end
    @variable(m, it, start= it0)
    @variable(m, iint, start= it0)

    @variable(m, csp_s, start = csp_avg)
    @variable(m, csn_s, start = csn_avg)

    @constraint(m,  5*(csp_s - csp_avg) + Rpp*it/F/Dp/ap/lp == 0)
    @constraint(m,  5*(csn_s - csn_avg) - Rpn*iint/F/Dn/an/lnn == 0)


    @variable(m, 0<=theta_p<=1, start=theta_p_guess)
    @variable(m, 0<=theta_n<=1, start= theta_n_guess)

    @constraint(m,  theta_p*cspmax == csp_s)
    @constraint(m,  theta_n*csnmax == csn_s)

    @variable(m, Up, start= Up_guess)
    @variable(m, phi_p, start= Up_guess)
    @variable(m, Un, start= Un_guess)
    @variable(m, phi_n, start= Un_guess)
    @NLconstraint(m,  (cspmax-csp_s)^(0.5)*csp_s^(0.5)*sinh(0.5*F/R/T*(phi_p-Up)) - (it/ap/F/lp/(2*kp*ce^(0.5))) == 0)
    @NLconstraint(m,  (csnmax-csn_s)^(0.5)*csn_s^(0.5)*sinh(0.5*F/R/T*(phi_n-Un+(delta_sei/Kappa_sei+Rsei)*it/an/lnn)) + iint/an/F/lnn/(2*kn*ce^(0.5)) == 0)

    @NLconstraint(m, Up == 7.49983 - 13.7758*theta_p^0.5 + 21.7683*theta_p - 12.6985*theta_p^1.5 + 0.0174967/theta_p -0.41649*theta_p^(-0.5) -
	0.0161404*exp(100*theta_p-97.1069) + 0.363031 * tanh(5.89493 *theta_p -4.21921))
    @NLconstraint(m, Un == 9.99877 -9.99961*theta_n^0.5 -9.98836*theta_n + 8.2024*theta_n^1.5  + 0.23584/theta_n -2.03569*theta_n^(-0.5) -
         1.47266*exp(-1.14872*theta_n+2.13185)- 9.9989*tanh(0.60345*theta_n-1.58171) )


    @variable(m, isei, start =  an*lnn*ksei*exp(-1*F/R/T*(Un_guess - Urefs + delta_sei/Kappa_sei*it0/an/lnn)))
    @NLconstraint(m, 1e4*(-isei+an*lnn*ksei*exp(-1*F/R/T*(phi_n - Urefs + (delta_sei/Kappa_sei+Rsei)*it/an/lnn))) == 0)
    @constraint(m, -iint + it -isei == 0)
    if mode == 1
        @constraint(m,  it == value)
    elseif mode == 2
        @constraint(m,  phi_p - phi_n == value)
    elseif mode == 3
        @constraint(m,  it*(phi_p-phi_n) == value)
    end
    JuMP.optimize!(m)

    iint0 = JuMP.value(iint)
    csp_s0 = JuMP.value(csp_s)
    csn_s0 = JuMP.value(csn_s)
    phi_p0 = JuMP.value(phi_p)
    phi_n0 = JuMP.value(phi_n)
    pot0 = phi_p0 - phi_n0
    it0 = JuMP.value(it)
    isei0 = JuMP.value(isei)
    return csp_s0, csn_s0, iint0, phi_p0, phi_n0, pot0, it0, isei0
end








end 