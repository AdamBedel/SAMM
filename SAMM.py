""" The world's best SAMM code."""
from __future__ import annotations
import numpy as np
import scipy.constants as scipyc
from scipy.integrate import odeint, solve_ivp


from src import objects, circuits, physics

__author__ = "Adam Bedel"
__date__ = "September 16, 2025"

class sim():
    def __init__(self, args: objects.args = objects.args.default(), t = np.linspace(0, 150e-9, 1500)):
        self.recompile(args, t)
    
    def recompile(self, args: objects.args = objects.args.default(), t = np.linspace(0, 250e-9, 2500)):
        self.args = args
        A_gas = 2 * (1 - args.f_tritium) + 3 * args.f_tritium
        N_gas_ititial = args.prefill_density * physics.gasVolumeInitial(args) * 1000 * scipyc.N_A / A_gas
        self.y: objects.State = objects.State(
            circ = args.circ_init,
            eg = (3/2) * scipyc.k * args.prefill_temperature * N_gas_ititial,
            Nd = N_gas_ititial * (1 - args.f_tritium),
            Nt = N_gas_ititial * args.f_tritium,
            rg = args.rg,
            rl = physics.shellPositionsInitial(args),
            vl = np.zeros(args.n_shells)
        )
        self.t = t
    
    def run(self):
        temp, info = odeint(self._ode, self.y.flatten(), self.t, args=(self.args,), full_output=True)
        temp_array = objects.vector_to_dataclass(temp.T, objects.State, type(self.args.circ_init))
        self.solution = objects.stack_states(temp_array)
        self.I = np.array([self.solution.circ[s].I for s in range(self.t.shape[0])])
        self.V = np.array([self.solution.circ[s].V for s in range(self.t.shape[0])])

        return self.solution
    
    def run_with_diffusion(self):
        # initial state
        y = self.y.flatten()
        tgrid = self.t
        Nt = len(tgrid)

        # initialize B(x)
        self.B = np.zeros(self.y.rl.shape[0] + 1)

        # ---- NEW: storage for B ----
        self.B_history = np.zeros((Nt, self.B.size))
        self.B_history[0] = self.B.copy()
        # ----------------------------

        # storage
        Y = np.zeros((Nt, len(y)))
        Y[0] = y

        for n in range(Nt-1):
            t0 = tgrid[n]
            t1 = tgrid[n+1]
            dt = t1 - t0

            # (1) Update magnetic field via explicit diffusion
            if n == 0:
                self.B = np.zeros(self.y.rl.shape[0] + 1)
                self.B_history[1] = self.B
                self.args.btheta = self.B

            else:
                _, _, self.B = physics.diffusion_nonuniform(np.append(state_obj.rg, state_obj.rl), self.args.btheta, dt, self.args.mat.res / scipyc.mu_0, scipyc.mu_0 * state_obj.circ.I / (2 * scipyc.pi * state_obj.rl[-1]))
                self.args.btheta = self.B[-1]
                self.B_history[n+1] = self.B[-1].copy()
            
            # (2) Advance SAMM
            y = self.step_ode(y, t0, t1, self.args)
            state_obj = objects.vector_to_dataclass(y, objects.State, type(self.args.circ_init))
            Y[n+1] = y

        # convert back into dataclass structure
        temp_array = objects.vector_to_dataclass(Y.T, objects.State, type(self.args.circ_init))
        self.solution = objects.stack_states(temp_array)
        self.I = np.array([self.solution.circ[s].I for s in range(self.t.shape[0])])
        self.V = np.array([self.solution.circ[s].V for s in range(self.t.shape[0])])

        return self.solution
    
    def step_ode(self, y, t0, t1, args):
        sol = solve_ivp(
            fun=lambda t, y: self._ode(y, t, args),
            t_span=(t0, t1),
            y0=y,
            method='RK45',
            max_step=(t1 - t0),
            rtol=1e-6,
            atol=1e-9,
        )
        return sol.y[:, -1]    # return the final state

    
    def plot_powerbalance(self):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        ax.fill_between(self.t * 1e9, self.solution.rg * 1e3, self.solution.rl[:, -1] * 1e3, color="#BE6400", label="Liner Radius [mm]")
        ax.plot(self.t * 1e9, self.I / 1e7, c='#32CD32', label = "Current [MA / 10]")

        ax.set_xlim((0, 140))
        ax.set_ylim((0, 8))
        ax.set_xlabel("Time [ns]")
        ax.set_ylabel("Amplitude (see legend)")

        ax.tick_params(axis='both', direction='in')
        ax.minorticks_on()
        ax.tick_params(axis='both', which='minor', direction='in')

        ax.legend()
        return ax

    
    def plot_trajectory(self, ymax=3, show_vmax=False, curScale='MA', title=None):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()

        ax.plot(self.t * 1e9, self.solution.rg * 1e3, color="k")
        ax.plot(self.t * 1e9, self.solution.rl[:, -1] * 1e3, color="k")
        ax.fill_between(self.t * 1e9, self.solution.rg * 1e3, self.solution.rl[:, -1] * 1e3, color="gray", alpha=0.5, label="Liner [mm]")
        if curScale == 'kA':
            ax.plot(self.t * 1e9, self.solution.circ.I / 1e6, c='red', label = "Current [MA]")
        else:
            ax.plot(self.t * 1e9, self.I / 1e7, c='red', label = "Current [MA / 10]")
        # ax.axvspan(0, r_0, alpha=0.9, color='pink', label='Plasma Region')
        # ax.axvline(r_rc, linestyle='--', c='gray', label="Return Radius")

        ax.set_xlim((0, self.t[-1] * 1e9))
        ax.set_ylim(0, ymax)
        ax.set_xlabel("Time [ns]")
        ax.set_ylabel("Amplitude (see legend)")

        ax.tick_params(axis='both', direction='in')
        ax.minorticks_on()
        ax.tick_params(axis='both', which='minor', direction='in')

        if title:
            ax.set_title(r"$ \Pi =$" + f"{physics.pi(self.t, self.solution, self.args) : .2f}")

        if show_vmax:
            ax.set_title(r"$ \Pi =$" + f"{physics.pi(self.t, self.solution, self.args) : .2f}, " + "$ KE @ CR20 = $" + f"{physics.KE(0, self.solution, self.args) * 1e-3 : .2f}" + " KJ")

        ax.legend()
        return ax

    @staticmethod
    # def _ode(y_vector: np.ndarray, t: float, *args_tuple: float):
    def _ode(y_vector, t, args):

        y: objects.State = objects.vector_to_dataclass(y_vector, objects.State, circ_type=type(args.circ_init))

        # Seatbelt
        if (y.rg < 1e-10):
            print("Imploded way too far!! Things are going to break")
            return

        mass_liner_shell = physics.linerVolumeInitial(args) * args.mat.rho / args.n_shells
        interfaces = np.append(y.rg, y.rl)  # length rl + 1
        pressure = np.zeros(y.rl.shape[0] + 2)  # length rl + 2


        # b_theta = scipyc.mu_0 * y.circ.I / (2 * scipyc.pi * y.rl[-1]) * ((interfaces - y.rg) / (y.rl[-1] - y.rg)) ** args.mat.beta

        if args.btheta[-1] >= 0:
            b_theta = args.btheta
            pressure[1:] += b_theta**2 / (2 * scipyc.mu_0)
            pressure[0] += b_theta[0]**2 / (2 * scipyc.mu_0)
        else:
            b_theta = scipyc.mu_0 * y.circ.I / (2 * scipyc.pi * y.rl[-1]) * ((interfaces - y.rg) / (y.rl[-1] - y.rg)) ** args.mat.beta
            pressure[1:] += b_theta**2 / (2 * scipyc.mu_0)
            
        liner_pressure = physics.coldCurve(y, args)
        pressure[1:-1] += liner_pressure
        pressure[1:-1] += physics.artificial_viscosity(y, args)
        Bz_internal = args.Bz * (scipyc.pi * args.rg**2) / (scipyc.pi * y.rg**2)
        pressure[0] += Bz_internal ** 2 / (2 * scipyc.mu_0)
        pressure[0] += physics.gasPressure(y, args)

        vl_dot = (pressure[1:-1] - pressure[2:]) * physics.shellSAs(y, args) / (mass_liner_shell)
        vl_dot[-1] = (pressure[-2] - pressure[-1]) * physics.linerSA(y, args) / (mass_liner_shell / 2)
        vg_dot = (pressure[0] - pressure[1]) * physics.gasSA(y, args) / (mass_liner_shell / 2)

        # Gas energy balance
        P_pdv = - (4/3) * y.eg * y.vg / y.rg
        eg_dot = P_pdv + physics.P_ph(t, y, args) - (physics.P_brems(y, args) if args.brems else 0)

        # Nuclear physics
        Ndt_neut_dot = physics.reactionRate(y, args, "DT")
        Ndd_neut_dot = physics.reactionRate(y, args, "DDn")
        Ndd_prot_dot = physics.reactionRate(y, args, "DDp")

        # clamp
        # vl_dot[1:] = np.where(physics.shellPositionsInitial(args)[1:] - physics.shellPositionsInitial(args)[:-1] < (y.rl[1:] - y.rl[:-1]) * 2, vl_dot[:-1], vl_dot[1:])

        clamp = 2.0
        for i in np.where(y.rl[1:] - y.rl[:-1] > clamp * (physics.shellPositionsInitial(args)[1:] - physics.shellPositionsInitial(args)[:-1]))[0]:
            if y.vl[i+1] > y.vl[i]:
                y.vl[i+1] = y.vl[i]

        return objects.State(
            circ = y.circ.evolve(t, y, args),
            eg = eg_dot, ## change in gas energy
            Nd = - Ndt_neut_dot - Ndd_neut_dot - Ndd_prot_dot,
            Nt = - Ndt_neut_dot,
            Ndd_neut = Ndd_neut_dot,
            Ndt_neut = Ndt_neut_dot,
            rg = y.vg,
            rl = y.vl,
            vl = vl_dot,
            vg = vg_dot
        ).flatten()