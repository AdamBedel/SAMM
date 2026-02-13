from dataclasses import dataclass
import numpy as np
import scipy.constants as scipyc

def artificial_viscosity(y, args):
    """
    Compute an artificial viscosity pressure term for shock-capturing.

    This implements a quadratic artificial viscosity that activates only in
    compressive regions (negative velocity gradient), using the shell densities
    as a scaling factor.

    Args:
        y (objects.State): Current simulation state.
        args (objects.args): Simulation configuration.

    Returns:
        numpy.ndarray: Artificial viscosity term per shell/interface (shape ~ n_shells).
    """
    v = np.append(y.vg, y.vl)
    dv = v[1:] - v[:-1]
    return args.aq**2 * shellDensities(y, args) * np.where(dv < 0, dv**2, 0)

def coaxInductance(h, rout, rin):
    """
    Compute the inductance of a coaxial geometry.

    Uses the standard expression for coax inductance per length integrated over
    a height h: L = (mu0 * h / (2*pi)) * ln(rout/rin).

    Args:
        h (float): Coaxial length/height [m].
        rout (float): Outer radius [m].
        rin (float): Inner radius [m].

    Returns:
        float: Inductance [H].
    """
    return scipyc.mu_0 * h / (2 * scipyc.pi) * np.log(rout/rin)

def coldCurve(y, args):
    """
    Compute the liner cold-curve pressure contribution.

    Uses the material cold-curve parameters (args.mat.cc) and the shell density
    ratio inferred from current shell volumes to return a pressure-like term
    for each shell.

    Args:
        y (objects.State): Current simulation state.
        args (objects.args): Simulation configuration.

    Returns:
        numpy.ndarray: Cold-curve pressure term per shell (shape n_shells).
    """

    mass_liner_shells: float = shellVolumesInitial(args) * args.mat.rho
    rho_ratio = (mass_liner_shells / shellVolumes(y, args)) / args.mat.rho

    # rho_ratio = shellDensities(y, args)

    # temp alias
    c = args.mat.cc
    
    return (
        (3/2) * c.a1
        * (rho_ratio**c.g1 - rho_ratio**c.g2)
        * (1 + (3/4) * (c.a2 - 4) * (rho_ratio**(2/3) - 1))
    )

def diffusion_nonuniform(x, phi0, tmax, D=1, right=1, C=0.99):
    """
    Solve a 1-D diffusion equation on a non-uniform grid using an explicit scheme.

    This function advances a diffusion-like field phi(x,t) over a grid `x` with
    spacing that may be non-uniform. The implementation uses a stability-limited
    timestep based on the smallest dx and applies boundary conditions at both ends.

    Notes:
        - The interior update is written in cylindrical form using fluxes
          proportional to r * dphi/dr.
        - Left boundary is reflecting (zero-gradient): phi[0] = phi[1].
        - Right boundary is Dirichlet: phi[-1] = right.

    Args:
        x (numpy.ndarray): Spatial grid (monotonic), length N.
        phi0 (numpy.ndarray): Initial condition phi(x, t=0), length N.
        tmax (float): Total integration time [s].
        D (float, optional): Diffusivity coefficient.
        right (float, optional): Right boundary value (Dirichlet).
        C (float, optional): Stability factor (< 1) for dt selection.

    Returns:
        tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]:
            (x, t, phi) where phi has shape (Nt, N).
    """

    N = len(x)
    
    # local dx
    dx = np.diff(x)  # length N-1
    
    # estimate min dx for stability
    dt = C * np.min(dx)**2 / (2*D)
    # print(dt)
    Nt = int(np.ceil(tmax / dt)) + 1
    t = np.linspace(0, tmax, Nt)

    # try forcing a single diffusion step instead
    # dt = tmax
    # Nt = 1
    # t = np.linspace(0, tmax, Nt)
    
    # storage
    phi = np.zeros((Nt, N))
    phi[0,:] = phi0
    
    # precompute dx_im1 and dx_i for interior points
    dx_im1 = x[1:-1] - x[:-2]   # length N-2
    dx_i   = x[2:]   - x[1:-1]  # length N-2
    
    # time stepping
    for n in range(Nt-1):
        u = phi[n]
        u_new = u.copy()
        
        # vectorized update for interior points
        # u_new[1:-1] = u[1:-1] + D*dt*(
        #     2 / (dx_im1 + dx_i) * ( (u[2:] - u[1:-1])/dx_i - (u[1:-1] - u[:-2])/dx_im1 )
        # )
        # r_i values
        r = x

        # old dx_im1, dx_i already computed
        # define half-node radii:
        r_iphalf = 0.5*(x[2:]   + x[1:-1])   # r_{i+1/2} for i=1..N-2
        r_imhalf = 0.5*(x[1:-1] + x[:-2])    # r_{i-1/2} for i=1..N-2

        # fluxes at half-nodes (F = r * dphi/dr)
        F_ip = r_iphalf * (u[2:]   - u[1:-1]) / dx_i
        F_im = r_imhalf * (u[1:-1] - u[:-2])  / dx_im1

        # central half-width Delta r at node i: r_{i+1/2} - r_{i-1/2}
        delta_r_cent = r_iphalf - r_imhalf   # = (dx_i + dx_im1)/2

        # finally the cylindrical update
        u_new[1:-1] = u[1:-1] + D*dt * (F_ip - F_im) / ( r[1:-1] * delta_r_cent )
        
        # boundary conditions
        # u_new[0] = 0.0   # all current trapped in liner
        u_new[0] = u_new[1]  # reflecting / zero gradient


        # u_new[-1] = u_new[-2]  # reflecting
        u_new[-1] = right
        
        phi[n+1] = u_new
    
    return x, t, phi

import numpy as np
import scipy.constants as scipyc

def diffusion_nonuniform_eta(x, phi0, tmax, T_profile, eta_of_T, right=1.0, C=0.99):
    """
    Same as diffusion_nonuniform, but with spatially varying diffusivity from a table:

        D(r) = eta(T(r)) / mu0

    The PDE solved is:
        dphi/dt = (1/r) d/dr ( r * D(r) * dphi/dr )

    Args:
        x (np.ndarray): radial grid, length N (monotonic).
        phi0 (np.ndarray): initial phi(x,0), length N.
        tmax (float): total integration time.
        T_profile (np.ndarray): temperature at each node, length N (or broadcastable to N).
        eta_of_T (callable): function returning resistivity eta [ohm-m] for T [K].
        right (float): Dirichlet BC at outer boundary: phi[-1] = right.
        C (float): stability factor (<1).

    Returns:
        (x, t, phi): phi has shape (Nt, N).
    """

    x = np.asarray(x, dtype=float)
    phi0 = np.asarray(phi0, dtype=float)
    T_profile = np.asarray(T_profile, dtype=float)

    N = len(x)
    if phi0.shape[0] != N:
        raise ValueError(f"phi0 length {phi0.shape[0]} must match x length {N}")
    if T_profile.shape[0] != N:
        raise ValueError(f"T_profile length {T_profile.shape[0]} must match x length {N}")

    #D from table
    mu0 = scipyc.mu_0
    eta_n = eta_of_T(T_profile)          # ohm-m, shape (N,)
    D_n = eta_n / mu0                    # m^2/s, shape (N,)

    # local dx
    dx = np.diff(x)                      # length N-1

    # worst case diffusivity
    Dmax = np.max(D_n)
    if Dmax <= 0:
        raise ValueError("Non-positive diffusivity encountered (check eta_of_T and units).")

    dt = C * np.min(dx)**2 / (2 * Dmax)
    Nt = int(np.ceil(tmax / dt)) + 1
    t = np.linspace(0.0, tmax, Nt)

    # storage
    phi = np.zeros((Nt, N), dtype=float)
    phi[0, :] = phi0

    # precompute geometry factors
    dx_im1 = x[1:-1] - x[:-2]           # length N-2
    dx_i   = x[2:]   - x[1:-1]          # length N-2

    r_iphalf = 0.5*(x[2:]   + x[1:-1])  # length N-2
    r_imhalf = 0.5*(x[1:-1] + x[:-2])   # length N-2
    delta_r_cent = r_iphalf - r_imhalf  # length N-2

    #face diffusivities
    # D_{i+1/2} between nodes i and i+1
    D_iphalf = 2*D_n[1:-1]*D_n[2:]  / (D_n[1:-1] + D_n[2:]  + 1e-300)
    D_imhalf = 2*D_n[1:-1]*D_n[:-2] / (D_n[1:-1] + D_n[:-2] + 1e-300)

    # time stepping
    for n in range(Nt - 1):
        u = phi[n]
        u_new = u.copy()

        # fluxes at half-nodes now include D:
        # F = r * D * dphi/dr
        F_ip = r_iphalf * D_iphalf * (u[2:]   - u[1:-1]) / dx_i
        F_im = r_imhalf * D_imhalf * (u[1:-1] - u[:-2])  / dx_im1

        # cylindrical update
        u_new[1:-1] = u[1:-1] + dt * (F_ip - F_im) / (x[1:-1] * delta_r_cent)

        # boundary conditions
        u_new[0]  = u_new[1]   # reflecting / zero gradient
        u_new[-1] = right      # Dirichlet

        phi[n+1] = u_new

    return x, t, phi



def gasPressure(y, args): 
    """
    Compute the fuel (gas) pressure assuming an ideal-gas-like relation.

    Uses P = (2/3) * E / V, where E is the fuel internal energy.

    Args:
        y (objects.State): Current simulation state.
        args (objects.args): Simulation configuration .

    Returns:
        float: Gas pressure [Pa].
    """
    return y.eg * (2/3) / gasVolume(y, args)

def gasSA(y, args): 
    """
    Compute the lateral surface area of the cylindrical fuel column.

    Args:
        y (objects.State): Current simulation state (expects y.rg).
        args (objects.args): Simulation configuration (expects args.h).

    Returns:
        float: Gas surface area [m^2].
    """
    return 2 * scipyc.pi * y.rg * args.h

def gasTemp(y, args): 
    """
    Compute the fuel temperature in Kelvin from internal energy.

    This assumes an ideal-gas-like relation using E = (3/2) N k T,
    with N inferred from deuterium/tritium particle bookkeeping.

    Args:
        y (objects.State): Current simulation state.
        args (objects.args): Simulation configuration.

    Returns:
        float: Fuel temperature [K].
    """
    return y.eg * (2/3) / (y.Nd + y.Nt * 2) / scipyc.k

def gasTempkev(y, args): 
    """
    Compute the fuel temperature in keV from internal energy.

    Converts Kelvin-equivalent temperature to energy units via e and 1e3.

    Args:
        y (objects.State): Current simulation state.
        args (objects.args): Simulation configuration.

    Returns:
        float: Fuel temperature [keV].
    """
    return y.eg * (2/3) / ((y.Nd + y.Nt) * 2) / scipyc.e / 1000

def gasVolume(y, args): 
    """
    Compute the current fuel volume assuming a cylinder.

    Args:
        y (objects.State): Current simulation state.
        args (objects.args): Simulation configuration.

    Returns:
        float: Fuel volume [m^3].
    """
    return scipyc.pi * y.rg**2 * args.h

def gasVolumeInitial(args): 
    """
    Compute the initial fuel volume from args assuming a cylinder.

    Args:
        args (objects.args): Simulation configuration.

    Returns:
        float: Initial fuel volume [m^3].
    """
    return scipyc.pi * args.rg**2 * args.h

def idealLinerPressure(y, args): #change once liner temp profile is added
    """
    Compute an ideal-gas-like liner pressure estimate at a fixed temperature.

    This uses P = n k T with T hard-coded to 300 K, and n derived from a liner
    number density model.

    Args:
        y (objects.State): Current simulation state.
        args (objects.args): Simulation configuration.

    Returns:
        float: Estimated liner pressure [Pa].
    """
    return numDensityLiner(y, args) * scipyc.k * 300  # assumes room temp

def KE(t, solution, args):
    """
    Estimate liner kinetic energy at 20× convergence (CR=20).

    This searches the solution trajectory for the first time index where the
    liner radius falls below the initial liner radius divided by 20, then
    computes KE = (1/2) m v^2 using a lumped liner mass estimate.

    Args:
        t (numpy.ndarray): Time array [s].
        solution (objects.StateSeries): Time series solution.
        args (objects.args): Simulation configuration.

    Returns:
        float: Estimated kinetic energy [J].

    Notes:
        As written, this function references `args.rl`, but args dataclass
        uses `rl0`. Ensure the mass and CR20 check use the intended variable.
    """
    mass = scipyc.pi * args.h * (args.rl ** 2 - args.rg ** 2) * args.mat.rho

    for idx, radius in enumerate(solution.rl):
        if radius <= args.rl / 20:
            CR_20_idx = idx
            break

    KE = 1/2 * mass * solution.vl[CR_20_idx] ** 2
    return KE

def linerSA(y, args): 
    """
    Compute the outer lateral surface area of the liner cylinder.

    Args:
        y (objects.State): Current simulation state.
        args (objects.args): Simulation configuration.

    Returns:
        float: Liner outer surface area [m^2].
    """
    return 2 * scipyc.pi * y.rl[-1] * args.h

def linerVolume(y, args):
    """ Returns either current or initial volume depending
    on if state or args were given as arguments. """
    """
    Compute the current liner volume assuming a cylindrical annulus.

    V = pi * h * (r_outer^2 - r_inner^2), with r_outer = y.rl[-1] and r_inner = y.rg.

    Args:
        y (objects.State): Current simulation state.
        args (objects.args): Simulation configuration.

    Returns:
        float: Current liner volume [m^3].
    """
    return ((scipyc.pi * args.h) *
            (y.rl[-1] ** 2 - y.rg ** 2))

def linerVolumeInitial(args):
    """ Returns either current or initial volume depending
    on if state or args were given as arguments. """
    """
    Compute the initial liner volume assuming a cylindrical annulus.

    V0 = pi * h * (rl0^2 - rg^2), with rl0 from args and inner radius rg from args.

    Args:
        args (objects.args): Simulation configuration.

    Returns:
        float: Initial liner volume [m^3].
    """
    return ((scipyc.pi * args.h) *
            (args.rl0 ** 2 - args.rg ** 2))

def numDensityLiner(y, args):
    """
    Estimate liner number density from liner mass and current liner volume.

    Computes a gram-based mass estimate using the initial liner volume and rho,
    then converts to number density using atomic mass `args.mat.a` and Avogadro's number.

    Args:
        y (objects.State): Current simulation state.
        args (objects.args): Simulation configuration.

    Returns:
        float: Estimated liner number density [1/m^3].
    """
    liner_mass_grams = linerVolumeInitial(args) * args.mat.rho * 1e3
    return liner_mass_grams * args.mat.a * scipyc.N_A / linerVolume(y, args)

def P_brems(y, args):
    """
    Compute bremsstrahlung power loss from the fuel region.

    Uses a threshold: if T_keV < 2 keV, returns 0. Otherwise evaluates a
    model of the form P ~ A * n^2 * sqrt(T) * V.

    Args:
        y (objects.State): Current simulation state.
        args (objects.args): Simulation configuration.

    Returns:
        float: Bremsstrahlung power loss [W].
    """
    if y.eg * (2/3) / ((y.Nd + y.Nt) * 2) / scipyc.e / 1000 < 2:
        return 0
    Abr = 1.57e-40
    val = Abr * ((y.Nt + y.Nd) / gasVolume(y, args)) ** 2 * np.sqrt(gasTemp(y, args)) * gasVolume(y, args)
    return val

def P_ph(t, y, args):
    """
    Compute preheat power deposition as a rectangular pulse.

    Returns constant power during the interval [ph_time, ph_time + ph_duration],
    with total energy equal to ph_energy.

    Args:
        t (float): Current simulation time [s].
        y (objects.State): Current simulation state.
        args (objects.args): Simulation configuration.

    Returns:
        float: Preheat power [W].
    """
    if t >= args.ph_time and t <= args.ph_time + args.ph_duration:
        return args.ph_energy / args.ph_duration
    return 0

def pi(t, solution, args):
    """
    Compute the dimensionless Pi parameter for the implosion.

    Uses the peak current time and a characteristic liner mass-per-length to form
    a dimensionless scaling parameter.

    Args:
        t (numpy.ndarray): Time array [s].
        solution (objects.StateSeries): Solution time series.
        args (objects.args): Simulation configuration.

    Returns:
        float: Dimensionless Pi parameter.

    Notes:
        As written, this function references `args.rl`, but args dataclass earlier
        used `rl0`. Ensure the intended variable is used consistently.
    """
    Ipeak = np.max(solution.circ.I)
    Ipeak_idx = np.argmax(solution.circ.I)
    mass = scipyc.pi * args.h * (args.rl ** 2 - args.rg ** 2) * args.mat.rho
    pi = scipyc.mu_0 * Ipeak ** 2 * t[Ipeak_idx] ** 2 / (4 * scipyc.pi * (mass / args.h) * args.rl ** 2)
    return pi

def reactivity(temp_kev, reaction):
    """
    Compute Maxwellian-averaged fusion reactivity <sigma v> for a given reaction.

    Uses fit coefficients stored in the Reaction class (DT, DDn, DDp) and evaluates
    a parameterized expression as a function of ion temperature in keV.

    Args:
        temp_kev (float): Ion temperature [keV].
        reaction (str): Reaction key ("DT", "DDn", or "DDp").

    Returns:
        float: Reactivity <sigma v> [m^3/s].
    """
    R = getattr(Reaction, reaction)()
    tkv = temp_kev

    num = R.c2 * tkv + R.c4 * tkv ** 2 + R.c6 * tkv ** 3
    den = 1 + R.c3 * tkv + R.c5 * tkv ** 2 + R.c7 * tkv ** 3

    c = 1 - num / den
    e = R.c0 / tkv ** (1/3)

    return R.c1 * c ** (-5/6) * e ** 2 * np.exp(-3 * c ** (1/3) * e)

def reactionRate(y, args, reaction):
    """
    Compute volumetric fusion reaction rate for a specified reaction channel.

    Uses <sigma v>(T) from `reactivity` and particle counts to form a rate per volume.

    Conventions:
        - DT: rate ~ Nd * Nt / V
        - DD: rate ~ 0.5 * Nd^2 / V (accounts for double counting)

    Args:
        y (objects.State): Current simulation state.
        args (objects.args): Simulation configuration.
        reaction (str): Reaction key ("DT", "DDn", or "DDp").

    Returns:
        float: Reaction rate [1/s].
    """
    sigma_v = reactivity(gasTempkev(y, args), reaction)
    if reaction == "DT":
        return sigma_v * y.Nd * y.Nt / gasVolume(y, args)
    if reaction == "DDn" or reaction == "DDp":
        return 0.5 * sigma_v * y.Nd * y.Nd / gasVolume(y, args)
    
    print("Undefined Reaction!! Uh-Oh")
    return np.inf


def shellPositionsInitial(args):
    """
    Compute initial liner shell interface radii.

    The shells are spaced uniformly in cross-sectional area between rg and rl0,
    resulting in radii proportional to sqrt(rg^2 + fraction*(rl0^2 - rg^2)).

    Args:
        args (objects.args): Simulation configuration.

    Returns:
        numpy.ndarray: Initial shell interface radii [m], shape (n_shells,).
    """
    return np.sqrt(args.rg*args.rg + (args.rl0*args.rl0 - args.rg*args.rg) * np.arange(1, args.n_shells+1) / args.n_shells)

def shellSAs(args_or_state, args): 
    """
    Compute shell lateral surface areas for each interface radius.

    Args:
        args_or_state: Object containing `rl`.
        args (objects.args): Simulation configuration.

    Returns:
        numpy.ndarray: Surface areas [m^2] for each shell radius.
    """
    return 2 * scipyc.pi * args.h * args_or_state.rl

def shellDensities(y, args): 
    """
    Compute current shell mass density for each shell.

    Uses fixed initial shell masses divided by current shell volumes.

    Args:
        y (objects.State): Current simulation state.
        args (objects.args): Simulation configuration.

    Returns:
        numpy.ndarray: Shell densities [kg/m^3], shape (n_shells,).
    """
    return shellVolumesInitial(args) * args.mat.rho / shellVolumes(y, args)

def shellVolumes(y, args):
    """
    Compute current shell volumes.

    Shell interfaces are [y.rg, y.rl[0], ..., y.rl[-1]].

    Args:
        y (objects.State): Current simulation state.
        args (objects.args): Simulation configuration.

    Returns:
        numpy.ndarray: Shell volumes [m^3], shape (n_shells,).
    """
    interfaces = np.append(y.rg, y.rl)
    return ((scipyc.pi * args.h) *
            (interfaces[1:] ** 2 - interfaces[:-1] ** 2))

def shellVolumesInitial(args):
    """
    Compute initial shell volumes for a cylindrical annulus discretization.

    Shell interfaces are [args.rg, shellPositionsInitial(args)].

    Args:
        args (objects.args): Simulation configuration.

    Returns:
        numpy.ndarray: Initial shell volumes [m^3], shape (n_shells,).
    """
    interfaces = np.append(args.rg, shellPositionsInitial(args))
    return ((scipyc.pi * args.h) *
            (interfaces[1:] ** 2 - interfaces[:-1] ** 2))

@dataclass
class Reaction:
    """
    Parameter set for fusion reactivity fit coefficients.

    The coefficients (c0..c7) define a temperature-dependent fit used by
    `reactivity(temp_kev, reaction)` to compute Maxwellian-averaged <sigma v>.

    Attributes:
        c0..c7 (float): Fit coefficients.
    """
    c0: float
    c1: float
    c2: float
    c3: float
    c4: float
    c5: float
    c6: float
    c7: float
    @classmethod
    def DT(cls):
        """
        Construct reactivity fit coefficients for the DT reaction channel.

        Returns:
            Reaction: Coefficient set for DT reactivity.
        """
        return cls(
            c0 = 6.661,
            c1 = 643.41e-22 * 0.98,
            c2 = 15.136e-3,
            c3 = 75.189e-3,
            c4 = 4.6064e-3,
            c5 = 13.5e-3,
            c6 = -0.10675e-3,
            c7 = 0.01366e-3,
        )
    
    @classmethod
    def DDn(cls):
        """
        Construct reactivity fit coefficients for the DDn channel.

        Returns:
            Reaction: Coefficient set for DDn reactivity.
        """
        return cls(
            c0 = 6.2696,
            c1 = 3.5741e-22,
            c2 = 5.8577e-3,
            c3 = 7.6822e-3,
            c4 = 0,
            c5 = -0.002964e-3,
            c6 = 0,
            c7 = 0,
        )

    @classmethod
    def DDp(cls):
        """
        Construct reactivity fit coefficients for the DDp channel.

        Returns:
            Reaction: Coefficient set for DDp reactivity.
        """
        return cls(
            c0 = 6.2696,
            c1 = 3.7212e-22,
            c2 = 3.4127e-3,
            c3 = 1.9917e-3,
            c4 = 0,
            c5 = 0.010506e-3,
            c6 = 0,
            c7 = 0,
        )
