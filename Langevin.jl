using DifferentialEquations


function drift!(du,u,p,t)
    P, gamma_r, kappa, gamma_A, n0, alpha = p
    du[1] = gamma_r*(2*u[2] - n0)*u[1] + gamma_r*u[2] - kappa*u[1]
    du[2] = P*(n0-u[2]) - gamma_r*(2*u[2] - n0)*u[1] - gamma_r*u[2] - gamma_A*u[2]
    end


function diffusion!(du, u, p, t)
    n_a, n_e, φ = u
    P, gamma_r, kappa, gamma_A, n0, alpha = p
    # Noise terms F_a and F_e with correlations D_aa, D_ae, D_ee
    D_aa = 0.5 * (kappa * n_a + gamma_r * n_e + gamma_r * n0 * n_a)
    D_ae = -0.5 * (gamma_r * n0 * n_a + gamma_r * n_e)
    D_ee = 0.5 * (P * (n0 - n_e) + (gamma_r + gamma_A) * n_e + gamma_r * n0 * n_a)

    # Define the diffusion matrix for the noise terms
    du[1,1] = sqrt(2*D_aa)  # noise term for n_a
    du[1,2] = -sqrt(2*abs(D_ae))  # noise term coupling between n_a and n_e
    du[2,1] = -sqrt(2*abs(D_ae))  # same as D_ae
    du[2,2] = sqrt(2*D_ee)  # noise term for n_e
    return du
end



function diffusion_avg!(du, u, p, t)
    P, gamma_r, kappa, gamma_A, n0, alpha,n_a, n_e = p
    # Noise terms F_a and F_e with correlations D_aa, D_ae, D_ee
    D_aa = 0.5 * (kappa * n_a + gamma_r * n_e + gamma_r * n0 * n_a)
    D_ae = -0.5 * (gamma_r * n0 * n_a + gamma_r * n_e)
    D_ee = 0.5 * (P * (n0 - n_e) + (gamma_r + gamma_A) * n_e + gamma_r * n0 * n_a)
    
   
    # Define the diffusion matrix for the noise terms
    du[1,1] = sqrt(2*D_aa)  # noise term for n_a
    du[1,2] = -sqrt(2*abs(D_ae))  # noise term coupling between n_a and n_e
    du[2,1] = -sqrt(2*abs(D_ae))  # same as D_ae
    du[2,2] = sqrt(2*D_ee)  # noise term for n_e
    return du
end


function reflect_condition(u, t, integrator)
    return minimum(u[1:2]) < 0
end

function reflect_affect!(integrator)
    integrator.u[1:2] .= abs.(integrator.u[1:2])
end

function clamp_affect!(integrator)
    integrator.u[1] = max(integrator.u[1],0)
    integrator.u[2] = max(integrator.u[2],0)
end

reflect_cb = DiscreteCallback(reflect_condition, reflect_affect!, save_positions = (false,false))

clamp_cb = DiscreteCallback(reflect_condition, clamp_affect!, save_positions = (false,false))



upper_cond(u, t, int) = u[2] > int.p[5]   # read n0 from the parameter set

function upper_affect!(int)
    n0 = int.p[5]
    int.u[2] = min(int.u[2],n0)             # reflect at n0  (mirror image)
end

upper_cb = DiscreteCallback(upper_cond, upper_affect!)

# ---- wrap them together ----------------------------------------------------
cb_limiter = CallbackSet(reflect_cb, upper_cb)

cb_limiter_clamp = CallbackSet(clamp_cb, upper_cb)



function accumulate_integrator!(u, t, integrator,stat)
    dt       = integrator.dt              # size of *last* step
    stat[1] += u[1]*dt
    stat[2] += u[2]*dt
    stat[3] += u[1]^2*dt
    stat[4] += u[2]^2*dt
    stat[5] += u[1]*u[2]*dt
    stat[6] += dt
    return nothing                        # no allocation, nothing is saved
end

