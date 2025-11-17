using Random
using ProgressMeter
using Statistics
using LinearAlgebra

"""
    tau_leaping(eq::StochasticRateEquation, nsteps::Int; eps=0.03, N0=nothing, max_tau=Inf, fixed_tau=nothing)

Solve a stochastic rate equation using the tau-leaping method.

# Arguments
- `eq::StochasticRateEquation`: The stochastic rate equation to solve
- `nsteps::Int`: Number of steps to take
- `eps::Float64=0.03`: Error control parameter for tau selection (smaller = more accurate but slower)
- `N0=nothing`: Initial population vector (optional)
- `max_tau=Inf`: Maximum allowed time step
- `fixed_tau=nothing`: Use a fixed time step instead of adaptive (optional)

# Returns
- Dictionary containing results similar to the GFRM method
"""
function tau_leaping(eq::StochasticRateEquation, nsteps::Int; eps=0.03, N0=nothing, max_tau=Inf, fixed_tau=nothing)
    # Initialize population vector
    N = zeros(Float64, size(eq.A, 1))
    if !isnothing(N0)
        N .= N0
    end
    
    # Initialize averages array (for first and second moments)
    avgs = init_avgs(size(eq.A,1))
    
    # Initialize rate vector
    a = zeros(Float64, size(eq.A, 2))
    
    # Pre-allocate k_events vector
    k_events = zeros(Int, size(eq.A, 2))
    
    # Run the main tau-leaping loop
    tau_leap_loop!(avgs, N, a, k_events, eq.A, nsteps, eq.parameters, eq.rate_func, eq.population_func, eps, max_tau, fixed_tau)
    #println(avgs)
    return Dict("averages" => avgs[1:end-1]/avgs[end], "t_end" => avgs[end])
end

"""
    tau_leaping_intra(eq::StochasticRateEquation, nsteps::Int; eps=0.03, N0=nothing, max_tau=Inf, fixed_tau=nothing)

Solve a stochastic rate equation using the tau-leaping method and record time series data.

# Arguments
- Similar to tau_leaping, but records population time series

# Returns
- Dictionary with averages, time series, and time vector
"""
function tau_leaping_intra(eq::StochasticRateEquation, nsteps::Int; eps=0.03, N0=nothing, max_tau=Inf, fixed_tau=nothing)
    # Initialize population vector
    N = zeros(Float64, size(eq.A, 1))
    if !isnothing(N0)
        N .= N0
    end
    
    # Initialize averages array
    avgs = init_avgs(size(eq.A,1))
    
    # Initialize rate vector
    a = zeros(Float64, size(eq.A, 2))
    
    # Pre-allocate k_events vector
    k_events = zeros(Int, size(eq.A, 2))
    
    # Allocate arrays for recording time series
    data = zeros(Float64, (length(N), nsteps))
    timevec = zeros(Float64, nsteps)
    
    # Run the main tau-leaping loop with time series recording
    tau_leap_loop_intra!(avgs, N, a, k_events, eq.A, nsteps, eq.parameters, eq.rate_func, eq.population_func, data, timevec, eps, max_tau, fixed_tau)
    return Dict("averages" => avgs[1:end-1]/avgs[end], "t_end" => avgs[end], "time_series" => data, "times" => timevec)
end

"""
    tau_leaping_out(eq::StochasticRateEquation, nsteps::Int, fout::Function; eps=0.03, N0=nothing, max_tau=Inf, fixed_tau=nothing)

Solve a stochastic rate equation using the tau-leaping method and record outcoupled events.

# Arguments
- Similar to tau_leaping, plus:
- `fout::Function`: Function to handle outcoupling events

# Returns
- Dictionary with averages and outcoupled event data
"""
function tau_leaping_out(eq::StochasticRateEquation, nsteps::Int, fout::Function; eps=0.03, N0=nothing, max_tau=Inf, fixed_tau=nothing)
    # Initialize population vector
    N = zeros(Float64, size(eq.A, 1))
    if !isnothing(N0)
        N .= N0
    end
    
    # Initialize averages array
    avgs = init_avgs(size(eq.A,1))
    
    # Initialize rate vector
    a = zeros(Float64, size(eq.A, 2))
    
    # Pre-allocate k_events vector
    k_events = zeros(Int, size(eq.A, 2))
    
    # Determine output type from fout function
    output_type = Float64
    for k in 1:size(eq.A, 2)
        out, out_result = fout(N, k)
        if out_result
            output_type = typeof(out)
            break
        end
    end
    
    # Allocate arrays for recording outcoupling events
    data_out = zeros(output_type, nsteps)
    timevec_out = zeros(Float64, nsteps)
    outcouple_duration = zeros(Float64, nsteps+2)
    outcouple_nemitter = zeros(BigInt, nsteps+2)
    
    # Track outcoupling events
    outcouple_counter = Ref(1)
    
    function outcouple_handler!(data_out, timevec_out, outcouple_duration, N, avgs, k_events, tau, outcouple_counter)
        # Accumulate time duration for potential outcoupling events
        outcouple_duration[end-1] += tau * max(1, N[1])
        
        # For each reaction channel
        for (k, k_count) in enumerate(k_events)
            if k_count > 0
                # Process events for this channel
                # Check if this is an outcoupling event
                output, event_result = fout(N, k)
                
                # If this is an outcoupling event
                if event_result && outcouple_counter[] <= length(data_out)
                    data_out[outcouple_counter[]] = output
                    timevec_out[outcouple_counter[]] = avgs[end]
                    
                    # Set decay duration for this event to the accumulated value
                    outcouple_duration[outcouple_counter[]] = outcouple_duration[end-1]
                    # Reset accumulator after outcoupling
                    outcouple_duration[end-1] = 0
                    
                    outcouple_counter[] += 1
                end
                
                # Reset accumulator if spontaneous emission (k==5) perturbs the phase
                if k == 5
                    outcouple_duration[end-1] = 0
                end
                
            end
        end
    end
    
    # Run the main tau-leaping loop with outcoupling recording
    tau_leap_loop_out!(avgs, N, a, k_events, eq.A, nsteps, eq.parameters, eq.rate_func, eq.population_func, 
                      data_out, timevec_out, outcouple_duration, outcouple_nemitter, outcouple_counter,
                      outcouple_handler!, eps, max_tau, fixed_tau)
    
    # Calculate how many actual outcoupling events were recorded
    actual_events = min(outcouple_counter[] - 1, nsteps)
    
    return Dict(
        "averages" => avgs[1:end-1]/avgs[end],
        "t_end" => avgs[end],
        "out_series" => data_out[1:actual_events],
        "out_times" => timevec_out[1:actual_events],
        "out_decay" => outcouple_duration[1:actual_events],
        "emitter" => outcouple_nemitter[1:actual_events]
    )
end
"""
    select_tau(a, N, A, eps, max_tau)

Choose a τ for explicit tau-leaping.

* Reactions with zero propensity are ignored (they cannot fire anyway),
  so they no longer clamp τ to a conservative fallback.
* If **all** propensities are zero the function returns `max_tau`
  (nothing happens until something turns on).

Arguments
---------
a        – vector of propensities (length M)
N        – vector of current populations (length S)
A        – S × M stoichiometry / population-change matrix
eps      – leap-condition tolerance (0.01–0.05 is typical)
max_tau  – user-supplied upper bound for τ
"""
function select_tau(
        a,
        N,
        ν,
        ε=0.01)

    S, R = size(ν)
    @assert length(a) == R
    @assert length(N) == S
    a₀ = sum(a)

    # ==== 1. Species based bound ==== ---------------------------------------
    μ  = ν * a                        # expected change of each species
    σ2 = (ν .^ 2) * a                 # variance of change  (⟨ΔXᵢ²⟩)

    g  = max.(1.0, N)                 # CGP’s “guard” for small populations
    τ_sp = Inf
    for i in 1:S
        if μ[i] ≠ 0
            τ_sp = min(τ_sp,  ε * g[i] / abs(μ[i]))
        end
        if σ2[i] ≠ 0
            τ_sp = min(τ_sp, (ε * g[i])^2 / σ2[i])
        end
    end
    # ==== 3. Final step size ==== -------------------------------------------
    #τ = min(τ_sp, τ_pr)
    return τ_sp
end


"""
    tau_leap_events!(N, a, A, tau, rate_func, params)

Calculate the number of events for each reaction channel during a tau-leaping step.
"""
function tau_leap_events!(N, a, A, tau, rate_func, params, k_events=nothing)
    # Update reaction rates
    
    # Calculate expected number of reactions during tau
    expected_events = a .* tau
    
    # Generate Poisson random numbers for each reaction channel
    if isnothing(k_events)
        k_events = [Poisson(expected) for expected in expected_events]
    else
        for i in eachindex(expected_events)
            k_events[i] = Poisson(expected_events[i])
        end
    end
    
    return k_events
end

"""
    tau_leap_loop!(avgs, N, a, k_events, A, nsteps, params, rate_func, population_func, eps, max_tau, fixed_tau)

Main loop for the tau-leaping algorithm.
"""
function tau_leap_loop!(avgs, N, a, k_events, A, nsteps, params, rate_func, population_func, eps, max_tau, fixed_tau)
    prog = Progress(100, desc="Tau-Leaping Progress: ", barglyphs=BarGlyphs("[=> ]"), barlen=50)
    i_mod = nsteps ÷ 100
    
    t_total = 0.0
    
    for i in 1:nsteps
        if i % i_mod == 0
            next!(prog)
        end
        rate_func(a, N, params)
    
        # Select tau (time step)
        if isnothing(fixed_tau)
            # Adaptive tau selection
            tau = select_tau(a, N, A, eps)
        else
            # Fixed tau
            tau = fixed_tau
        end
        
        # Calculate events for this step
        tau_leap_events!(N, a, A, tau, rate_func, params, k_events)
        
        # Update population for each reaction channel
        for j in 1:length(k_events)
            population_func(N, A, j, tau/ length(k_events), params; k_events=k_events[j])
        end
        
        # Update time and record statistics
        t_total += tau
        time_averages!(avgs, N, tau)
    end
    
    return avgs
end

"""
    tau_leap_loop_intra!(avgs, N, a, k_events, A, nsteps, params, rate_func, population_func, data, timevec, eps, max_tau, fixed_tau)

Tau-leaping loop that also records time series data.
"""
function tau_leap_loop_intra!(avgs, N, a, k_events, A, nsteps, params, rate_func, population_func, data, timevec, eps, max_tau, fixed_tau)
    prog = Progress(100, desc="Tau-Leaping Progress: ", barglyphs=BarGlyphs("[=> ]"), barlen=50)
    i_mod = nsteps ÷ 100
    
    t_total = 0.0
    
    for i in 1:nsteps
        if i % i_mod == 0
            next!(prog)
        end
        rate_func(a, N, params)
    
        # Select tau (time step)
        if isnothing(fixed_tau)
            # Adaptive tau selection
            tau = select_tau(a, N, A, eps)
        else
            # Fixed tau
            tau = fixed_tau
        end
        
        # Calculate events for this step
        tau_leap_events!(N, a, A, tau, rate_func, params, k_events)
        
        # Update population for each reaction channel
        for j in 1:length(k_events)
            population_func(N, A, j, tau/ length(k_events), params; k_events=k_events[j])
        end
        
        # Update time and record statistics
        t_total += tau
        time_averages!(avgs, N, tau)
        
        # Record time series data
        save_series!(data, timevec, N, i, t_total)
    end
    
    return avgs
end

"""
    tau_leap_loop_out!(avgs, N, a, k_events, A, nsteps, params, rate_func, population_func, data_out, timevec_out, outcouple_duration, outcouple_nemitter, outcouple_counter, outcouple_handler!, eps, max_tau, fixed_tau)

Tau-leaping loop that records outcoupling events.
"""
function tau_leap_loop_out!(avgs, N, a, k_events, A, nsteps, params, rate_func, population_func, 
                           data_out, timevec_out, outcouple_duration, outcouple_nemitter, outcouple_counter,
                           outcouple_handler!, eps, max_tau, fixed_tau)
    prog = Progress(100, desc="Tau-Leaping Progress: ", barglyphs=BarGlyphs("[=> ]"), barlen=50)
    i_mod = nsteps ÷ 100
    
    t_total = 0.0
    
    # Continue until we've collected enough outcoupling events or reached max steps
    step_count = 0
    max_steps = nsteps * 100  # Safety limit to prevent infinite loops
    
    while outcouple_counter[] <= nsteps && step_count < max_steps
        step_count += 1
        
        if step_count % i_mod == 0
            next!(prog)
        end
        rate_func(a, N, params)
    
        # Select tau (time step)
        if isnothing(fixed_tau)
            # Adaptive tau selection
            tau = select_tau(a, N, A, eps)
        else
            # Fixed tau
            tau = fixed_tau
        end
        
        # Calculate events for this step
        tau_leap_events!(N, a, A, tau, rate_func, params, k_events)
        
        # Handle outcoupling events
        outcouple_handler!(data_out, timevec_out, outcouple_duration, N, avgs, k_events, tau, outcouple_counter)
        
        # Update population for each reaction channel
        for j in 1:length(k_events)
            population_func(N, A, j, tau / length(k_events), params; k_events=k_events[j])
        end
        
        # Update time and record statistics
        t_total += tau
        time_averages!(avgs, N, tau)
    end
    
    return avgs
end

# Poisson random number generator
function Poisson(lambda)
    if lambda <= 0
        return 0
    elseif lambda < 20
        # Direct method for small lambda
        L = exp(-lambda)
        k = 0
        p = 1.0
        
        while p > L
            k += 1
            p *= rand()
        end
        
        return k - 1
    else
        # Normal approximation for large lambda
        return round(Int, (Normal(lambda, sqrt(lambda))))
    end
end

# Normal distribution random number generator using Box-Muller transform
function Normal(mu, sigma)
    u1 = rand()
    u2 = rand()
    z = sqrt(-2.0 * log(u1)) * cos(2π * u2)
    return mu + sigma * z
end 



"""
    tau_gillespie(N, a, ν; ε = 0.03, params = nothing)

Return an admissible leap‐size τ for explicit τ-leaping according to the
Cao–Gillespie–Petzold (2006) rule.

Arguments
---------
* `N`      : Vector of current populations (length S).
* `a`      : Vector of propensities aᵣ(N) (length R).
* `ν`      : Stoichiometry matrix of size (S × R); ν[i,r] = change of species i
             when reaction r fires once.
* `ε`      : User–chosen accuracy parameter (default 0.01)..

Returns
-------
A positive float – the largest time step that satisfies the τ-leaping
conditions for both species and propensities.
"""
function tau_leaping_dt(
        N,
        a,
        ν;
        ε = 0.01,)

    S, R = size(ν)
    @assert length(a) == R
    @assert length(N) == S
    a₀ = sum(a)

    # ==== 1. Species based bound ==== ---------------------------------------
    μ  = ν * a                        # expected change of each species
    σ2 = (ν .^ 2) * a                 # variance of change  (⟨ΔXᵢ²⟩)

    g  = max.(1.0, N)                 # CGP’s “guard” for small populations

    τ_sp = Inf
    for i in 1:S
        if μ[i] ≠ 0
            τ_sp = min(τ_sp,  ε * g[i] / abs(μ[i]))
        end
        if σ2[i] ≠ 0
            τ_sp = min(τ_sp, (ε * g[i])^2 / σ2[i])
        end
    end
    println("Species based bound τ_sp = $τ_sp")

    return τ_sp
end

