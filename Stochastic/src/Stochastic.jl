module Stochastic
    using Random
    using ProgressMeter

    export StochasticRateEquation,
    gfrm_out,gfrm_intra,gfrm,
    tau_leaping,tau_leaping_intra,tau_leaping_out,tau_leaping_dt

    include("GFRM.jl")
    include("TauLeaping.jl")
    include("SignalProcessing.jl")

end # module Stochastic
