module assignedValues
__precompile__() 

export nHours_Horizon, state_dim, inner_state_dim, nweeks, Totaltime, TotalHours, dt_FR, Nt_FR_hour, Nt_FR, TIME_FR, Ncp, Ncn, Nsei, Ncum, F, R, T, M_sei, Kappa_sei, rho_sei, ksei, Urefs, Rsei, area, cspmax, csnmax, lp, lnn, Rpp, Rpn, ce, ep, en, ap, an, kp, kn, Dn, Dp, TC, Qmax, P_nominal,
    differential_vars,
    capacity_retire,
    soc_min,
    soc_max,
    soc_min_stop,
    soc_max_stop,
    maxC,
    umax,
    umin,
    soc,
    csp_avg0,
    csn_avg0,
    delta_sei0,
    u_start



    


nweeks = 52   

TotalHours = ((nweeks*7+1)*24)                  
Totaltime = TotalHours*60*60                    

dt_FR = 2                                       
Nt_FR_hour = round(Int, 3600/dt_FR)            
Nt_FR = round(Int, Totaltime/dt_FR)+1           
TIME_FR = 0:dt_FR:Totaltime                    





Ncp = 2 
Ncn = 2 
Nsei = 3
Ncum = 4






F=96487
R=8.3143
T=298.15
M_sei=0.073
Kappa_sei=5e-6
rho_sei=2.1e3
ksei=1.5e-12
Urefs = 0.4
Rsei = 0.01
area = 0.3108
cspmax=10350
csnmax=29480
lp=6.521e-5
lnn=2.885e-5
Rpp=1.637e-7
Rpn=3.596e-6
ce=1042
ep = 1-0.52
en = 1-0.619
ap=3*ep/Rpp
an=3*en/Rpn
kp=1.127e-7/F
kn=8.696e-7/F
Dn=8.256e-14
Dp=1.736e-14
TC = 2.3/area
Qmax = TC
P_nominal = TC*3.1



state_dim = 6
inner_state_dim = Ncp+Ncn+4+Nsei+Ncum



differential_vars = trues(Ncp+Ncn+4+Nsei+Ncum)
differential_vars[Ncp] = false
differential_vars[Ncp+Ncn] = false
differential_vars[Ncp+Ncn+1] = false
differential_vars[Ncp+Ncn+2] = false
differential_vars[Ncp+Ncn+3] = false
differential_vars[Ncp+Ncn+4] = false
differential_vars[Ncp+Ncn+5] = false
differential_vars[Ncp+Ncn+6] = false



V_max = 3.65
V_min = 2.0


capacity_retire = 0.8
soc_min = 0.5
soc_max = 0.5
soc_min_stop = 0.1
soc_max_stop = 0.9



maxC = 10
umax = [P_nominal*maxC, P_nominal*maxC]
umin = [0, -P_nominal*maxC]



soc = 0.5
csp_avg0 = cspmax*1 - csnmax*soc*lnn*en/lp/ep           
csn_avg0 = csnmax*soc
delta_sei0 = 1e-10
u0 = zeros(Ncp+Ncn+4 + Nsei+Ncum)

u0[1:Ncp] .= csp_avg0
u0[(Ncp+1):(Ncp+Ncn)] .= csn_avg0
u0[Ncp+Ncn+7] = delta_sei0              #delta_sei

u_start = deepcopy(u0)




nHours_Horizon = 1

end 