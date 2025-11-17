# Modeling nanolaser quantum noise with Master equations, Markov chains and langevin stochastic differential equations
An implementation of various techniques for calculating quantum noise in lasers used in [paperlink](paperlink). For necessary context of the implementation, see the paper. The majority of methods are written in the programming language Julia and some in python. A similar implementation (without langevin rate equations and master equation) is also available at: [https://github.com/mabuni1998/GFRM-laser-quantum-phase-noise](https://github.com/mabuni1998/GFRM-laser-quantum-phase-noise). 

# Installation and usage

## Markov chains

The Gillespie method and tau-leaping approaches are implemented in the module `Stochastic`, which is installed by downloading the folder `Stochastic` and add the Stochastic module to the current Julia environment with the command (press ] to get to the package manager):
```julia
pkg> dev Stochastic\\
```

Subsequently, the package is loaded by calling.
```julia
using Stochastic
```

We also use the following packages (JLD2 for loading files, UnPack for unpacking paramteers, and DifferentialEquations.jl). Thes are installed using:

```julia
pkg> add JLD2, UnPack, DifferentialEquations
```

## Langevin stochastic differential equations

The langevin stochastic differential equations are solved using [DifferentialEquations.jl](https://docs.sciml.ai/DiffEqDocs/stable/) and are setup in the file `Langevin.jl`, which should be downloaded and installed using:

```julia
include("Langevin.jl")
```

## Master equation
The master equations are solved using the [Permutational Invariant Quantum Solver](https://qutip.readthedocs.io/en/latest/guide/guide-piqs.html). The implementation can be found in the file `PIQS_laser.py`.

# Examples of usage
In the following, we showcase how the different approaches are used. The parameters used (same as in paper) are stored in the file `parameters_betascaling.jld`.

In julia we load the parameters:


```julia
parameters = load_object("parameters_betascaling.jld")
@unpack g, kappa, gamma_As, gamma_D, n0s, Ps = parameters
```


Subsequently, we can pick the emitter number $n_0$ and pump value $P$ with:

```julia
idx_n0s,idx_Ps = 1,10
P = parameters[:P] = Ps[idx_n0s+1][idx_Ps+1]
n0 = parameters[:n0] = n0s[idx_n0s+1]
gamma_A = parameters[:gamma_A] = gamma_As[idx_n0s+1]

gamma_r = 4*g^2 ./ (kappa+gamma_A+gamma_D + P)
```

In python, we can similarly load the parameters and choose the index to pick the parameters of the simulation. This is contained in the file `PIQS_laser.py`.

## Steady state values

In many of the simulations, we use the steady state value of the rate equations to better pick the time scales of the simulation. One can analytically solve the rate equations considered in the paper with:

```julia
function steady_state_values(p)
    P, gamma_r, kappa, gamma_A, n0 = p
    gt = gamma_r + gamma_A

    A = 2*gamma_r*kappa
    B= -P*n0*gamma_r - gamma_r^2*n0 + gamma_r*gt*n0 + P*kappa + gt*kappa
    C = -P*n0*gamma_r
    na_ss = 1/(2*A)*(-B + sqrt(B^2-4*A*C))
    ne_ss = na_ss*(gamma_r*n0+kappa)/(gamma_r*(2*na_ss+1))

    return na_ss,ne_ss
end
```

To find the analytical value of the g2-function we can use the following function which also returns the relaxation frequency, which is used to estimate the simulation time.

```julia
function analytical_langevin(params)
    P, gamma_r, kappa, gamma_A, n0,na,ne = params  # for consistency with other functions
    
    g_ee = P .+ (gamma_r + gamma_A) .+ 2 .* gamma_r .* na
    g_ep = gamma_r .* (2 * ne .- n0)
    g_pe = gamma_r .* (2 * na .+ 1)
    g_pp = kappa .- gamma_r .* (2 * ne .- n0)
    G = g_ee .+ g_pp
    D_ee = 1/2 .* (P .* (n0 .- ne) .+ gamma_r .* n0 .* na .+ (gamma_r + gamma_A) .* ne)
    D_pp = 1/2 .* (gamma_r .* na .* n0 .+ gamma_r .* ne .+ kappa .* na)
    D_ep = 1/2 .* (-gamma_r .* na .* n0 .- gamma_r .* ne)
    D_pe = D_ep
    wr = sqrt.(Complex.(g_ep .* g_pe .+ g_ee .* g_pp))

    dnp2  = 1/G*((1+g_ee^2/wr^2)*D_pp+g_pe^2/wr^2*D_ee+2*g_pe*g_ee/wr^2*D_pe)
    g2 = (dnp2-na)/na^2+1

    return g2,wr
end
```


## Markov chains - Gillespie's method and tau-leaping
We assume the parameters were loaded as in the above and the module `Stochastic` was installed. We load the module and collect the parameters in a list:
```julia
using Stochastic
params = [P,gamma_r,kappa,gamma_A,n0]
#Find steady state
na_ss,ne_ss = steady_state_values(params)
N_ss = [na_ss,ne_ss]
params_ss = [P,gamma_r,kappa,gamma_A,n0,na_ss,ne_ss]
```

In the code, one defines the different events and their effect on the populations by giving a matrix `A`. The matrix should have dimensions (`rows` $\times$ `columns`), where `rows` is the number of stochastic variables (here two: na, ne) and `columns` is the number of events (here 6).
```julia
#Define matrix A, which defines the population changes for each event.
A =[-1 -1  0  1  1  0; #Photon population change
     0  1 -1 -1 -1  1] #Emitter population change
```

Then, a function that gives the rates at each timestep should be defined. Here, `a` is a vector of length `columns` that contains the rates, `N` is a vector of length `rows` containing the stochastic variables, and params are any parameters necessary for the simulation.
```julia
#Define how the six rates are updated (see eq. 1 and 2 in the main text of the paper)
function update_rates!(a,N,params)
    P,gamma_r,kappa,gamma_A,n0 = params

    a[1] = kappa*N[1]                   #Cavity loss
    a[2] = gamma_r*N[1]*(n0-N[2])  #Stimulated absorption
    a[3] = gamma_A*N[2]                   #Non-radiative decay
    a[4] = gamma_r*N[2]*N[1]              #Stimulated emission
    a[5] = gamma_r*N[2]                   #Spontaneous emission
    a[6] = P*(n0 - N[2])     #Pump-event
    return
end
```

We then provide a function that updates the population (this function is run at each timestep). `k` here denotes which event (index of column) happened, and the population is changed according to that. For the tau-leaping we need to ensure non-negative populations and to clamp to emitter number to be maximum $n_0$.
```julia
function update_population!(N, A, k, dt, params;k_events=1)
    # Update the number of photons and emitters
    for i in 1:length(N)-1
        N[i] += A[i,k]*k_events
    end
    N[1] = max(0.0, N[1])  # Photon number cannot be negative
    n0 = params[5]  # n0 is the 5th parameter
    if N[2]> n0
        "Clamping"
    end
    N[2] = clamp(N[2], 0.0, n0)
    return
end

```
Combining all of this, we can define a stochastic rate equation problem:

```julia
#Define stochastic rate equation problem
prob = StochasticRateEquation(A,params,update_rates!,update_population!)    
```

This problem can be solved using either the Gillespie method or the tau-leaping method.

For tau-leaping, we need to pick dt well, which can be done using the function `tau_leaping_dt`, which picks dt according to Eq. 19 in the paper.

```julia
N_ss = [na_ss, ne_ss]
rates = zeros(6)
update_rates!(rates, [na_ss, ne_ss], params)
epsilon = 0.01
dt_characteristic_tau = tau_leaping_dt(N,rates,A;ε=epsilon)
```

We then pick the number of simulation steps based on the simulation time.

```julia
_,omega_r = analytical_langevin(params_ss)
T_sim = 100*5000/omega_r
n_steps_tau = round(Int,T_sim/dt_characteristic_tau)
```

Finally, we solve using the tau-leaping approach

```julia
result_tau = tau_leaping(prob, n_steps_tau; fixed_tau=dt_characteristic_tau,N0=[round(Int,na_ss), round(Int,ne_ss)])
t_end = result_tau["t_end"]
pops = result_tau["averages"]
na_tau = pops[1]
ne_tau = pops[2]
g2_tau = (pops[4] - pops[1])/pops[1]^2
```            
      

For gillespie's method we pick the number of time steps based on the characteristic timestep of the gillespie method which is the half the maximum of the inverse of the rates (found empirically):

```julia
dt_characteristic = 1/max(rates...)/2
steps =round(Int,T_sim/dt_characteristic)
```

The problem is then solved using the Gillespie method by calling `gfrm`:

```julia
result_out = gfrm(prob, steps;N0=[round(Int,na_ss), round(Int,ne_ss) ,0])
t_end = result_out["t_end"]
pops = result_out["averages"]
na_gfrm = pops[1]
ne_gfrm = pops[2]
g2_gfrm = (pops[4] - pops[1])/pops[1]^2
```

## Langevin stochastic rate equations

We start by loading the definition of the relevant functions for setting up the stochastic differential equation:

```julia
include("Langevin.jl")
```

We then determine the simulation time from the relaxation frequency as well as the initial state of the simulation (steady state values):

```julia
_,omega_r = analytical_langevin(params_ss)
T_sim_langevin = 5000/omega_r
tspan = (0.0, T_sim_langevin)
u0 = [na_ss, ne_ss] 
```

We then set up the stochastic differential equation problem using the predefined `drift!` and `diffusion_avg!` function which implement the ODE and the white noise (diffusion terms) - see `Langevin.jl`. We also include the callback function, which clamps the population to positive values at all times.

```julia
prob_sde = SDEProblem(drift!, diffusion_avg!, u0, tspan, params_ss,callback=cb_limiter_clamp,noise_rate_prototype=zeros(2, 2))
```

Because we will often take a very long series where we do not want to sae every step but only calculate the averages we also define a vector of `stats` where all the statistics are accumulated over the simulation. This is also done using a call back function.

```julia
stats = zeros(Float64, 6)  # Initialize statistics array
cb = FunctionCallingCallback((x,y,z)->accumulate_integrator!(x,y,z,stats),func_everystep = true)  
```


Finally, we can run the solver and get the stats afterwards:
```julia
sol = solve(prob_sde, EM(), dt=dt,callback=cb,
save_everystep=false,
save_start=false,
save_end=false,
dense=false)

na_langevin = stats[1] / stats[end]
ne_langevin = stats[2] / stats[end]
na2_langevin = stats[3] / stats[end]
ne2_langevin = stats[4] / stats[end]
g2_langevin = na2s[run] / nas[run]^2        
```
