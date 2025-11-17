"""
Permutational invariance simulation for a single Pump value.

Usage:
    python permutational_invariance_simulation_single.py <P_idx> [n0_idx] [output_dir]

Arguments:
    P_idx: Index of the P value to simulate (required)
    n0_idx: Index of the n0 value (default: 1)
    output_dir: Output directory for results (default: ./data/)
"""

import sys
import os
from qutip import *
from qutip.piqs.piqs import *
import numpy as np


def rate_equation(t, u, p):
    """
    Rate equation for the system.

    Parameters:
    - u: state vector [u[0], u[1]]
    - p: parameter vector [g, kappa, gamma_A, gamma_D, n0, P]
    - t: time (required by solve_ivp but not used in autonomous system)
    """
    g = p[0]
    kappa = p[1]
    gamma_A = p[2]
    gamma_D = p[3]
    n0 = p[4]
    P = p[5]

    gamma_r = 4*g**2 / (kappa + gamma_A + gamma_D + P)

    du = np.zeros(2)
    du[0] = gamma_r*(2*u[1] - n0)*u[0] + gamma_r*u[1] - kappa*u[0]
    du[1] = P*(n0 - u[1]) - gamma_r*(2*u[1] - n0)*u[0] - gamma_r*u[1] - gamma_A*u[1]

    return du


def rate_equation_for_fsolve(u, p):
    """Wrapper for fsolve (doesn't need time argument)."""
    return rate_equation(0, u, p)


def steady_state(parameters):
    """
    Find the steady state of the system.

    Parameters:
    - parameters: dictionary with keys 'g', 'kappa', 'gamma_A', 'gamma_D', 'n0', 'P'

    Returns:
    - sol: solution array [u[0], u[1]] at steady state
    """
    from scipy.optimize import fsolve

    g = parameters['g']
    kappa = parameters['kappa']
    gamma_A = parameters['gamma_A']
    gamma_D = parameters['gamma_D']
    n0 = parameters['n0']
    P = parameters['P']

    if P > 0.1:
        u0 = np.array([100, n0/2])
    else:
        u0 = np.array([n0/2, n0/2])
    p = [g, kappa, gamma_A, gamma_D, n0, P]

    # Find steady state using fsolve (finds where du/dt = 0)
    sol = fsolve(rate_equation_for_fsolve, u0, args=(p,), xtol=1e-8)

    return sol


def det_Hil(na):
    """Determine Hilbert space size based on expected photon number."""
    pop = 100
    k = int(na)
    if k == 0:
        return 10
    else:
        while pop > 10**(-12):
            pop = np.exp(k*np.log(na)-na-k*np.log(k)+k)
            k += 1
    return k


import h5py

def load_jld2_dict(filename):
    """Load a JLD2 file containing a Dict into a Python dictionary."""
    f = h5py.File(filename, "r")

    # Get the single stored object
    obj = f['single_stored_object']
    data = obj[()]

    # Dereference to get the dictionary data
    ref = data[0]
    dict_data = f[ref]
    dict_content = dict_data[()]

    # Build Python dictionary
    parameters = {}

    for entry_ref in dict_content:
        entry = f[entry_ref]
        entry_data = entry[()]

        # Get key (as string) and value reference
        key = entry_data['first'].decode('utf-8')  # Convert bytes to string
        value_ref = entry_data['second']

        # Dereference the value
        if isinstance(value_ref, h5py.h5r.Reference):
            value_dataset = f[value_ref]
            value = value_dataset[()]

            # If it's an array of references (like 'Ps'), dereference each element
            if isinstance(value, np.ndarray) and value.dtype == object:
                dereferenced_array = []
                for item in value:
                    if isinstance(item, h5py.h5r.Reference):
                        dereferenced_array.append(f[item][()])
                    else:
                        dereferenced_array.append(item)
                value = dereferenced_array

            parameters[key] = value

    return parameters, f  # Return file handle to keep it open



def main():
    # Parse command line arguments
    P_idx = 10 #Index for the pump value
    n0_idx = 1 #Index for the emitter number n0
    
    parameters, f = load_jld2_dict("parameters_betascaling.jld")

    # Extract relevant parameters
    g = parameters['g']
    kappa = parameters['kappa']
    gamma_D = parameters['gamma_D']
    gamma_A = parameters["gamma_A"] = parameters['gamma_As'][n0_idx]
    n0 = parameters["n0"] = parameters["n0s"][n0_idx]
    Ps = parameters['Ps'][n0_idx]

    P = parameters["P"] = Ps[P_idx]

    print(f"\nRunning simulation for:")
    print(f"  P_idx = {P_idx}")
    print(f"  P = {P:.6f}")
    print(f"  n0_idx = {n0_idx}")
    print(f"  n0 = {n0}")
    
    # TLS parameters
    n_tls = n0
    N = n_tls
    system = Dicke(N=n_tls)
    [jx, jy, jz] = jspin(N)
    jp = jspin(N, "+")
    jm = jp.dag()

    # Set up system
    system.hamiltonian = 0 * jz
    system.collective_dephasing = gamma_D
    system.pumping = P
    system.emission = gamma_A
    D_tls = system.liouvillian()

    # Calculate steady state from rate equations
    print("Calculating rate equation steady state...")
    sol = steady_state(parameters)
    nphot_guess = det_Hil(sol[0])+5
    print(f"  Rate eq: n_phot={sol[0]:.4f}, n_atom={sol[1]:.4f}")
    print(f"  Hilbert space size for photons: {nphot_guess}")

    # Create photon operators
    a = destroy(nphot_guess)

    # Interaction Hamiltonian
    h_int = g * tensor(a, jp) + g * tensor(a.dag(), jm)
    c_ops_phot = [np.sqrt(kappa) * a]
    D_phot = liouvillian(0 * a.dag()*a, c_ops_phot)

    # Build total Liouvillian
    nds = num_dicke_states(n_tls)
    id_tls = to_super(qeye(nds))
    id_phot = to_super(qeye(nphot_guess))

    D_int = -1j * spre(h_int) + 1j * spost(h_int)
    D_tot = D_int + super_tensor(D_phot, id_tls) + super_tensor(id_phot, D_tls)

    # Find steady state
    print("Calculating QuTiP steady state...")
    #rho_ss = steadystate(D_tot, method="iterative-bicgstab", maxiter=10000,
    #                       atol=1e-12)
    rho_ss = steadystate(D_tot, method="iterative-gmres",atol=1e-22)

    # Calculate expectation values
    nemitter = tensor(qeye(nphot_guess), jp*jm)
    jpjm_ss = expect(nemitter, rho_ss)

    nphot_tot = tensor(a.dag()*a, qeye(nds))
    nphot_ss = expect(nphot_tot, rho_ss)

    # Calculate g2
    a_tot = tensor(a, qeye(nds))
    ad_tot = tensor(a.dag(), qeye(nds))
    out = expect(ad_tot*ad_tot*a_tot*a_tot, rho_ss)
    g2 = out / nphot_ss**2

    print("\n" + "="*60)
    print(f"Results for P={P:.6f}:")
    print(f"  Atom population - Rate eq: {sol[1]:.4f}, QuTiP: {jpjm_ss:.4f}")
    print(f"  Photon number - Rate eq: {sol[0]:.4f}, QuTiP: {nphot_ss:.4f}")
    print(f"  g2: {g2:.4f}")
    print("="*60 + "\n")

    # Store result
    result = {
        'P': P,
        'nphot_ss': nphot_ss,
        'natom_ss': jpjm_ss,
        'g2': g2,
        'nphot_rate_eq': sol[0],
        'natom_rate_eq': sol[1]
    }

    print("\nSimulation completed successfully!")


if __name__ == "__main__":
    main()
