using Stochastic
using JLD2
using UnPack
using Statistics
using Random
using ProgressMeter  # Add this for progress bars

using Revise
includet("init_stochastic.jl")

# Load parameters
parameters = load_object("parameters_betascaling.jld")
@unpack g, kappa, gamma_As, gamma_D, n0s, Ps = parameters



# Process command line arguments
if length(ARGS) == 0
    println("No arguments given")
    i = Int(41*0+1)
    output_dir = "./data"
elseif length(ARGS) == 1
    i = parse(Int64, ARGS[1])
    output_dir = "./data"
else
    i = parse(Int64, ARGS[1])
    output_dir = joinpath("./data", ARGS[2])
    
    # Create output directory if it doesn't exist
    if !isdir(output_dir)
        println("Creating output directory: $output_dir")
        mkpath(output_dir)
    else
        println("Using existing output directory: $output_dir")
    end
end

# Extract parameters based on index
idx_n0s, idx_Ps = divrem(i-1, length(Ps[1]))
P = parameters[:P] = Ps[idx_n0s+1][idx_Ps+1]
n0 = parameters[:n0] = n0s[idx_n0s+1]
gamma_A = parameters[:gamma_A] = gamma_As[idx_n0s+1]

println("Running simulation for P = $P, n0 = $n0, job index = $i, output directory = $output_dir")

# Create reusable parameter string for filenames
param_str = "nena_g$(g)_k$(kappa)_ga$(gamma_A)_gd$(gamma_D)_n0$(n0)_P$(P)"

# Calculate steady state values
gamma_r = 4*g^2 ./ (kappa+gamma_A+gamma_D + P)
params = [P, gamma_r, kappa, gamma_A, n0, 0]
na_ss, ne_ss = steady_state_values(params)
params_ss = [P, gamma_r, kappa, gamma_A, n0, 0, na_ss, ne_ss]

# Configure which simulations to run
calc_gfrm = false
calc_tau = false
calc_langevin_ss = false
calc_langevin_ss_clamp = false
calc_langevin = true
calc_langevin_k_transformed = false
calc_langevin_k_transformed_ss = false

# Number of runs for statistics
N_runs = 5
dt_dividers = [1,2]
dt_divider_la = [1,2]

# Calculate characteristic time scales
rates = zeros(6)
update_rates!(rates, [na_ss, ne_ss], params)
dt_characteristic = 1/max(rates...)/2

max_steps = Int(2e10)
factor = 1 / (gamma_r*ne_ss*(na_ss+1)/(gamma_A*ne_ss+P*(n0-ne_ss)+kappa*na_ss))
_,tmax_gf = estimate_langevin_dt(params_ss)
#lower_limit = maximum(1 ./ params[1:5])
#tmax_gf = max(tmax_gf, lower_limit)
steps = min(max(round(Int,100*tmax_gf/dt_characteristic),round(Int,factor*10_000_000)),max_steps)
# Set the time step for tau-leaping

function pick_epsilon(ne_ss)
    # Function to pick epsilon based on the steady state values
    if ne_ss < 2e3
        return 0.01
    elseif ne_ss < 2e5
        return 0.005
    else
        return 0.001
    end
end

epsilon = pick_epsilon(ne_ss)
prob = StochasticRateEquation(A, params, update_rates!, update_population!)
N = [na_ss, ne_ss]
dt_characteristic_tau = tau_leaping_dt(N,rates,A;ε=epsilon,params=params)

tmax_gfrm = max(round(Int,100*tmax_gf/dt_characteristic),round(Int,factor*10_000_000))*dt_characteristic


function compute_length_factor(ne)
    # Function to adjust the length factor based on ne_ss
    if 1 < ne < 1e1+1
        return 10
    else
        return 1
    end
end
length_factor = compute_length_factor(n0)  # Adjust this if you want to scale the time
# Run GFRM simulation if enabled       
if calc_gfrm
    # Arrays to store results from multiple runs
    nas = zeros(Float64, N_runs)
    nes = zeros(Float64, N_runs)
    g2s = zeros(Float64, N_runs)
    ne2s = zeros(Float64, N_runs)
    nenas = zeros(Float64, N_runs)
    tends = zeros(Float64, N_runs)
    
    println("Running GFRM simulations...")
    progress = Progress(N_runs, desc="GFRM runs: ", showspeed=true)
    
    for run in 1:N_runs
        result_out = gfrm(prob, steps;N0=[round(Int,na_ss), round(Int,ne_ss) ,0])
        t_end = result_out["t_end"]
        tends[run] = t_end
        
        pops = result_out["averages"]
        nas[run] = pops[1]
        nes[run] = pops[2]
        g2s[run] = (pops[4] - pops[1])/pops[1]^2
        ne2s[run] = pops[5]
        nenas[run] = pops[7] /(pops[1]*pops[2])
        
        next!(progress)
    end
    
    # Save results
    d = Dict{String,Any}()
    @pack! d = nas, nes, g2s, ne2s, tends, nenas
    
    t_end = mean(tends)

    savename = "gfrm_$(param_str).jld"
    save_object(joinpath(output_dir, savename), d)
    println("Saved GFRM results to $(joinpath(output_dir, savename))")
else
    # Try to load t_end from previously saved GFRM results
    savename_gf = "gfrm_$(param_str).jld"
    try
        d_gf = load_object(joinpath(output_dir, savename_gf))
        if haskey(d_gf, "tends")
            t_end = mean(d_gf["tends"])
        end
    catch
        println("No previous GFRM results found, using default t_end = $t_end")
    end
end
#a

#calc_tau = true
# Run tau leaping simulation if enabled
if calc_tau
    println("Running tau leaping simulations...")
    #p_dividers = Progress(length(dt_dividers), desc="Processing dt dividers: ", showspeed=true)
    
    for dt_divider in dt_dividers
        dt_tau = dt_characteristic_tau/dt_divider
        N_tau = min(round(Int, tmax_gfrm/dt_tau),max_steps)
        
        # Arrays to store results from multiple runs
        nas = zeros(Float64, N_runs)
        nes = zeros(Float64, N_runs)
        g2s = zeros(Float64, N_runs)
        ne2s = zeros(Float64, N_runs)
        nenas = zeros(Float64, N_runs)
        tends = zeros(Float64, N_runs)
        
        p_runs = Progress(N_runs, desc="  Tau dt=$(dt_divider) runs: ", showspeed=true)
        
        for run in 1:N_runs
            result_out = tau_leaping(prob, N_tau; fixed_tau=dt_tau,N0=[round(Int,na_ss), round(Int,ne_ss) ,0])
            #result_out = tau_leaping(prob, N_tau; eps=0.01,N0=[round(Int,na_ss), round(Int,ne_ss) ,0])
            pops = result_out["averages"]
            t_end_run = result_out["t_end"]
            tends[run] = t_end_run
            
            nas[run] = pops[1]
            nes[run] = pops[2]
            g2s[run] = (pops[4] - pops[1])/pops[1]^2
            ne2s[run] = pops[5]
            nenas[run] = pops[7]/(pops[1]*pops[2])
            println(pops[1])
            next!(p_runs)
        end
        
        # Save results
        d = Dict{String,Any}()
        @pack! d = nas, nes, g2s, ne2s, tends, nenas
        
        savename = "tau_dt$(dt_divider)_$(param_str).jld"
        save_object(joinpath(output_dir, savename), d)
        println("Saved tau dt$(dt_divider) results to $(joinpath(output_dir, savename))")
        
        #next!(p_dividers)
    end
end
#d = load_object("./data/"*"tau_dt$(1)_$(param_str).jld")
#mean((d["g2s"] ) .+ (1 .- d["nas"]) ./ d["nas"])
#(d["g2s"] ) .+ (1 .- d["nas"]) ./ d["nas"]
#mean(d["g2s"])
#mean(d["nas"])
#d["nas"]
#na_ss

# Run Langevin simulation if enabled
if calc_langevin
    println("Running Langevin simulations...")
    #p_dividers = Progress(length(dt_dividers), desc="Processing dt dividers: ", showspeed=true)
    
    for dt_divider in dt_divider_la
        dt_,tmax = estimate_langevin_dt(params_ss,epsilon=epsilon)
        dt = dt_/dt_divider

        tspan = (0.0, tmax*length_factor)
        
        # Arrays to store results from multiple runs
        nas = zeros(Float64, N_runs)
        nes = zeros(Float64, N_runs)
        na2s = zeros(Float64, N_runs)
        ne2s = zeros(Float64, N_runs)
        g2s = zeros(Float64, N_runs)
        nenas = zeros(Float64, N_runs)
    

        # Set up initial conditions
        u0 = [na_ss, ne_ss, 0.0]  # [photons, emitters, phase]
        
        # Define simulation problem
        prob_sde = SDEProblem(drift!, diffusion!, u0, tspan, params,callback=cb_limiter,noise_rate_prototype=zeros(3, 3))
        
        p_runs = Progress(N_runs, desc="  Langevin dt=$(dt_divider) runs: ", showspeed=true)
        
        for run in 1:N_runs
            # Solve SDE
            
           stats = zeros(Float64, 8)  # Initialize statistics array
            cb = FunctionCallingCallback((x,y,z)->accumulate_integrator!(x,y,z,stats),  # callback function;
                             func_everystep = true)  # call after every step

            sol = solve(prob_sde, EulerHeun(), dt=dt,callback=cb,
            save_everystep=false,
            save_start=false,
            save_end=false,
            dense=false)
            
            nas[run] = stats[1] / stats[end]
            nes[run] = stats[2] / stats[end]
            na2s[run] = stats[4] / stats[end]
            ne2s[run] = stats[5] / stats[end]
            g2s[run] = na2s[run] / nas[run]^2
            nenas[run] = stats[7] / stats[end] /(nas[run]*nes[run])

            next!(p_runs)
        end
        
        # Save results
        d = Dict{String,Any}()
        @pack! d = nas, nes, g2s, na2s, ne2s, nenas
        
        savename = "langevin_dt$(dt_divider)_$(param_str).jld"
        save_object(joinpath(output_dir, savename), d)
        println("Saved Langevin dt$(dt_divider) results to $(joinpath(output_dir, savename))")
        
        #next!(p_dividers)
    end
end

# Run Langevin simulation with steady state noise if enabled
if calc_langevin_ss_clamp
    println("Running Langevin SS Clamp simulations...")
    #p_dividers = Progress(length(dt_dividers), desc="Processing dt dividers: ", showspeed=true)
    
    for dt_divider in dt_divider_la
        dt_,tmax = estimate_langevin_dt(params_ss,epsilon=epsilon)
        dt = dt_/dt_divider

        tspan = (0.0, tmax*length_factor)
        
        # Arrays to store results from multiple runs
        nas = zeros(Float64, N_runs)
        nes = zeros(Float64, N_runs)
        na2s = zeros(Float64, N_runs)
        ne2s = zeros(Float64, N_runs)
        g2s = zeros(Float64, N_runs)
        nenas = zeros(Float64, N_runs)
    

        # Set up initial conditions
        u0 = [na_ss, ne_ss, 0.0]  # [photons, emitters, phase]
        
        # Define simulation problem
        prob_sde = SDEProblem(drift!, diffusion_avg!, u0, tspan, params_ss,callback=cb_limiter_clamp,noise_rate_prototype=zeros(3, 3))
        
        p_runs = Progress(N_runs, desc="  Langevin SS dt=$(dt_divider) runs: ", showspeed=true)
        
        for run in 1:N_runs
            # Solve SDE
           stats = zeros(Float64, 8)  # Initialize statistics array
            cb = FunctionCallingCallback((x,y,z)->accumulate_integrator!(x,y,z,stats),  # callback function;
                             func_everystep = true)  # call after every step

            sol = solve(prob_sde, EM(), dt=dt,callback=cb,
            save_everystep=false,
            save_start=false,
            save_end=false,
            dense=false)
            
            nas[run] = stats[1] / stats[end]
            nes[run] = stats[2] / stats[end]
            na2s[run] = stats[4] / stats[end]
            ne2s[run] = stats[5] / stats[end]
            g2s[run] = na2s[run] / nas[run]^2
            nenas[run] = stats[7] / stats[end] /(nas[run]*nes[run])

            next!(p_runs)
        end
        
        # Save results
        d = Dict{String,Any}()
        @pack! d = nas, nes, g2s, na2s, ne2s, nenas
        
        savename = "langevin_ss_clamp_dt$(dt_divider)_$(param_str).jld"
        save_object(joinpath(output_dir, savename), d)
        println("Saved Langevin SS Clamp dt$(dt_divider) results to $(joinpath(output_dir, savename))")
        
        #next!(p_dividers)
    end
end
#d = load_object("./data/"*"langevin_ss_clamp_dt$(1)_$(param_str).jld")
#mean((d["g2s"] ) .+ (1 .- d["nas"]) ./ d["nas"])
#(d["g2s"] ) .+ (1 .- d["nas"]) ./ d["nas"]
#mean(d["g2s"])
#mean(d["nas"])
#na_ss

# Run Langevin simulation with steady state noise if enabled
if calc_langevin_ss
    println("Running Langevin SS  simulations...")
    #p_dividers = Progress(length(dt_dividers), desc="Processing dt dividers: ", showspeed=true)
    
    for dt_divider in dt_divider_la
        dt_,tmax = estimate_langevin_dt(params_ss,epsilon=epsilon)
        dt = dt_/dt_divider

        tspan = (0.0, tmax*length_factor)
        
        # Arrays to store results from multiple runs
        nas = zeros(Float64, N_runs)
        nes = zeros(Float64, N_runs)
        na2s = zeros(Float64, N_runs)
        ne2s = zeros(Float64, N_runs)
        g2s = zeros(Float64, N_runs)
        nenas = zeros(Float64, N_runs)
    
        # Set up initial conditions
        u0 = [na_ss, ne_ss, 0.0]  # [photons, emitters, phase]
        
        # Define simulation problem
        prob_sde = SDEProblem(drift!, diffusion_avg!, u0, tspan, params_ss,callback=cb_limiter,noise_rate_prototype=zeros(3, 3))
        
        p_runs = Progress(N_runs, desc="  Langevin SS dt=$(dt_divider) runs: ", showspeed=true)
        
        for run in 1:N_runs
            # Solve SDE
           stats = zeros(Float64, 8)  # Initialize statistics array
            cb = FunctionCallingCallback((x,y,z)->accumulate_integrator!(x,y,z,stats),  # callback function;
                             func_everystep = true)  # call after every step

            sol = solve(prob_sde, EM(), dt=dt,callback=cb,
            save_everystep=false,
            save_start=false,
            save_end=false,
            dense=false)
            
            nas[run] = stats[1] / stats[end]
            nes[run] = stats[2] / stats[end]
            na2s[run] = stats[4] / stats[end]
            ne2s[run] = stats[5] / stats[end]
            g2s[run] = na2s[run] / nas[run]^2
            nenas[run] = stats[7] / stats[end] /(nas[run]*nes[run])
            
            next!(p_runs)
        end
        
        # Save results
        d = Dict{String,Any}()
        @pack! d = nas, nes, g2s, na2s, ne2s, nenas
        
        savename = "langevin_ss_dt$(dt_divider)_$(param_str).jld"
        save_object(joinpath(output_dir, savename), d)
        println("Saved Langevin SS dt$(dt_divider) results to $(joinpath(output_dir, savename))")
        
        #next!(p_dividers)
    end
end


# Run Langevin k-transformed simulation if enabled
if calc_langevin_k_transformed_ss
    println("Running Langevin K-transformed SS simulations...")
    #p_dividers = Progress(length(dt_dividers), desc="Processing dt dividers: ", showspeed=true)
    
    for dt_divider in dt_divider_la
        dt_,tmax = estimate_langevin_dt(params_ss,epsilon=epsilon)
        dt = dt_/dt_divider
        tspan = (0.0, tmax*length_factor)
        
        # Arrays to store results from multiple runs
        nas = zeros(Float64, N_runs)
        nes = zeros(Float64, N_runs)
        na2s = zeros(Float64, N_runs)
        ne2s = zeros(Float64, N_runs)
        g2s = zeros(Float64, N_runs)
        nenas = zeros(Float64,N_runs)

        D_aa = 0.5 * (kappa * na_ss + gamma_r * ne_ss + gamma_r * n0 * na_ss)
        D_ae = -0.5 * (gamma_r * n0 * na_ss + gamma_r * ne_ss)
        k = -D_ae/D_aa
        
        # Set up initial conditions - modified for k-transformed system
        u0 = [na_ss, ne_ss+k*na_ss, 0.0]
        
        # Define simulation problem
        prob_sde = SDEProblem(drift_z_avg!, diffusion_z_avg!, u0, tspan, params_ss, callback=cb_limiter_z)
        
        p_runs = Progress(N_runs, desc="  Langevin K dt=$(dt_divider) runs: ", showspeed=true)
        
        for run in 1:N_runs
           stats = zeros(Float64, 8)  # Initialize statistics array
            cb = FunctionCallingCallback((x,y,z)->accumulate_integrator!(x,y,z,stats),  # callback function;
                             func_everystep = true)  # call after every step

            sol = solve(prob_sde, EM(), dt=dt,callback=cb,
            save_everystep=false,
            save_start=false,
            save_end=false,
            dense=false)
            
            # Collect statistics
            nas[run] = stats[1] / stats[end]
            nes[run] = (stats[2] - k*stats[1]) / stats[end]
            na2s[run] = stats[4] / stats[end]
            ne2s[run] = (stats[5])^2 / stats[end]
            g2s[run] = na2s[run] / nas[run]^2
            nenas[run] = stats[7] / stats[end] /(nas[run]*nes[run])
            
            next!(p_runs)
        end
        
        # Save results
        d = Dict{String,Any}()
        @pack! d = nas, nes, g2s, na2s, ne2s, nenas
        
        savename = "langevin_k_ss_dt$(dt_divider)_$(param_str).jld"
        save_object(joinpath(output_dir, savename), d)
        println("Saved Langevin K-transformed dt$(dt_divider) results to $(joinpath(output_dir, savename))")
        
        #next!(p_dividers)
    end
end

println("Completed all simulations for job $i: P=$P, n0=$n0")
