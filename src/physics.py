from dataclasses import dataclass
import numpy as np
import scipy.constants as scipyc

def artificial_viscosity(y, args):
    v = np.append(y.vg, y.vl)
    dv = v[1:] - v[:-1]
    return args.aq**2 * shellDensities(y, args) * np.where(dv < 0, dv**2, 0)

def coaxInductance(h, rout, rin):
    return scipyc.mu_0 * h / (2 * scipyc.pi) * np.log(rout/rin)

def coldCurve(y, args):

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


def gasPressure(y, args): return y.eg * (2/3) / gasVolume(y, args)

def gasSA(y, args): return 2 * scipyc.pi * y.rg * args.h

def gasTemp(y, args): return y.eg * (2/3) / (y.Nd + y.Nt * 2) / scipyc.k

def gasTempkev(y, args): return y.eg * (2/3) / ((y.Nd + y.Nt) * 2) / scipyc.e / 1000

def gasVolume(y, args): return scipyc.pi * y.rg**2 * args.h

def gasVolumeInitial(args): return scipyc.pi * args.rg**2 * args.h

def idealLinerPressure(y, args):
    return numDensityLiner(y, args) * scipyc.k * 300  # assumes room temp

def KE(t, solution, args):
    mass = scipyc.pi * args.h * (args.rl ** 2 - args.rg ** 2) * args.mat.rho

    for idx, radius in enumerate(solution.rl):
        if radius <= args.rl / 20:
            CR_20_idx = idx
            break

    KE = 1/2 * mass * solution.vl[CR_20_idx] ** 2
    return KE

def linerSA(y, args): return 2 * scipyc.pi * y.rl[-1] * args.h

def linerVolume(y, args):
    """ Returns either current or initial volume depending
    on if state or args were given as arguments. """
    return ((scipyc.pi * args.h) *
            (y.rl[-1] ** 2 - y.rg ** 2))

def linerVolumeInitial(args):
    """ Returns either current or initial volume depending
    on if state or args were given as arguments. """
    return ((scipyc.pi * args.h) *
            (args.rl0 ** 2 - args.rg ** 2))

def numDensityLiner(y, args):
    liner_mass_grams = linerVolumeInitial(args) * args.mat.rho * 1e3
    return liner_mass_grams * args.mat.a * scipyc.N_A / linerVolume(y, args)

def P_brems(y, args):
    if y.eg * (2/3) / ((y.Nd + y.Nt) * 2) / scipyc.e / 1000 < 2:
        return 0
    Abr = 1.57e-40
    val = Abr * ((y.Nt + y.Nd) / gasVolume(y, args)) ** 2 * np.sqrt(gasTemp(y, args)) * gasVolume(y, args)
    return val

def P_ph(t, y, args):
    if t >= args.ph_time and t <= args.ph_time + args.ph_duration:
        return args.ph_energy / args.ph_duration
    return 0

def pi(t, solution, args):
    Ipeak = np.max(solution.circ.I)
    Ipeak_idx = np.argmax(solution.circ.I)
    mass = scipyc.pi * args.h * (args.rl ** 2 - args.rg ** 2) * args.mat.rho
    pi = scipyc.mu_0 * Ipeak ** 2 * t[Ipeak_idx] ** 2 / (4 * scipyc.pi * (mass / args.h) * args.rl ** 2)
    return pi

def reactivity(temp_kev, reaction):
    R = getattr(Reaction, reaction)()
    tkv = temp_kev

    num = R.c2 * tkv + R.c4 * tkv ** 2 + R.c6 * tkv ** 3
    den = 1 + R.c3 * tkv + R.c5 * tkv ** 2 + R.c7 * tkv ** 3

    c = 1 - num / den
    e = R.c0 / tkv ** (1/3)

    return R.c1 * c ** (-5/6) * e ** 2 * np.exp(-3 * c ** (1/3) * e)

def reactionRate(y, args, reaction):
    sigma_v = reactivity(gasTempkev(y, args), reaction)
    if reaction == "DT":
        return sigma_v * y.Nd * y.Nt / gasVolume(y, args)
    if reaction == "DDn" or reaction == "DDp":
        return 0.5 * sigma_v * y.Nd * y.Nd / gasVolume(y, args)
    
    print("Undefined Reaction!! Uh-Oh")
    return np.inf


def shellPositionsInitial(args):
    return np.sqrt(args.rg*args.rg + (args.rl0*args.rl0 - args.rg*args.rg) * np.arange(1, args.n_shells+1) / args.n_shells)

def shellSAs(args_or_state, args): return 2 * scipyc.pi * args.h * args_or_state.rl

def shellDensities(y, args): return shellVolumesInitial(args) * args.mat.rho / shellVolumes(y, args)

def shellVolumes(y, args):
    interfaces = np.append(y.rg, y.rl)
    return ((scipyc.pi * args.h) *
            (interfaces[1:] ** 2 - interfaces[:-1] ** 2))

def shellVolumesInitial(args):
    interfaces = np.append(args.rg, shellPositionsInitial(args))
    return ((scipyc.pi * args.h) *
            (interfaces[1:] ** 2 - interfaces[:-1] ** 2))

@dataclass
class Reaction:
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
