from abc import ABC, abstractmethod
from dataclasses import dataclass
import numpy as np
import scipy.constants as scipyc
from scipy.interpolate import interp1d

@dataclass
class abstractCircuit(ABC):
    """
    Abstract class for circuit drive models.

    Circuit objects are stored inside the global simulation 'state' as 'y.circ'
    and are advanced by calling 'evolve(...)', which returns the time-derivative
    values of the circuit state variables packaged in a circuit dataclass instance.

    Attributes:
        I (float): Circuit current through the load [A] (or dI/dt depending on how
            the circuit is implemented in the ODE system).
    """
    I: float  # current through load

    @classmethod
    @abstractmethod
    def evolve(cls, t, y, args):
        return cls(I = 0)

@dataclass
class voltageDrivenCircuit(abstractCircuit):
    """
    Circuit model driven by a prescribed voltage waveform.

    State variables include current I and an effective inductance L. The `evolve`
    method computes derivatives based on a time-dependent applied voltage and
    geometry-driven inductance change.

    Attributes:
        I (float): Circuit current through the load [A] (or dI/dt in derivative form).
        L (float): Effective load inductance [H] (or dL/dt in derivative form).
    """
    L: float = 0
    @classmethod
    def evolve(cls, t, y, args):
        """
        Compute derivatives for a voltage-driven circuit.

        Uses a prescribed voltage waveform V(t) and includes a geometry-driven
        inductance change term L_dot based on the liner motion.

        Notes:
            - This implementation treats L_dot as proportional to y.vl / y.rl[-1].
            - The returned fields are interpreted as time derivatives for coupling
            to the ODE system.

        Args:
            t (float): Current simulation time [s].
            y (objects.State): Current simulation state.
            args (objects.args): Simulation configuration and physical parameters.

        Returns:
            voltageDrivenCircuit: Circuit derivative object with fields (I=I_dot, L=L_dot).
        """
        L_dot = - scipyc.mu_0 * args.h / (2 *scipyc.pi) * y.vl / y.rl[-1]
        I_dot = (cls.explicit_voltage(t) - L_dot * y.circ.I) / y.circ.L
        return cls(
            I = I_dot,
            L = L_dot
        )
    
    @staticmethod
    def explicit_voltage(t_eval) -> float:
        """
        Evaluate the prescribed drive voltage waveform at a given time.

        The waveform is defined by a sampled cosine profile over a fixed time window,
        then interpolated to the requested evaluation time.

        Args:
            t_eval (float): Evaluation time [s].

        Returns:
            float: Applied drive voltage V(t_eval) [V].
        """
        t = np.linspace(0, 250e-9, 100)
        V_array = 275e3 * np.cos(t * scipyc.pi / 200e-9)
        return interp1d(t, V_array)(t_eval)

@dataclass
class LCCircuit(abstractCircuit):
    """
    Lumped-element LC circuit model with evolving inductance.

    This model advances current I and capacitor voltage V with a capacitor C,
    and includes an inductance change term L_dot driven by liner motion.

    Attributes:
        I (float): Circuit current [A] (or dI/dt in derivative form).
        L (float): Inductance [H] (or dL/dt in derivative form).
        V (float): Capacitor voltage [V] (or dV/dt in derivative form).
        C (float): Capacitance [F].
    """
    L: float
    V: float
    C: float = 0
    @classmethod
    def evolve(cls, t, y, args):
        """
        Compute derivatives for an LC circuit with time-varying inductance.

        Equations implemented:
            L_dot: geometry-driven inductance change from liner motion
            I_dot: (V - L_dot * I) / L
            V_dot: -I / C

        Args:
            t (float): Current simulation time [s].
            y (objects.State): Current simulation state.
            args (objects.args): Simulation configuration and physical parameters.

        Returns:
            LCCircuit: Circuit derivative object with fields (I=I_dot, L=L_dot, V=V_dot).
        """
        L_dot = - scipyc.mu_0 * args.h / (2 *scipyc.pi) * y.vl[-1] / y.rl[-1]
        I_dot = (y.circ.V - L_dot * y.circ.I) / y.circ.L
        V_dot = - y.circ.I / y.circ.C
        return cls(
            I = I_dot,
            L = L_dot,
            V = V_dot,
        )


@dataclass
class sineCircuit(abstractCircuit):
    """
    Prescribed sine-like current drive model.

    This model does not evolve a dynamic circuit state; instead, it imposes
    a current time dependence via a parameterized waveform. The `evolve` method
    returns the instantaneous time derivative of the current.

    Attributes:
        Ipeak (float): Peak current amplitude [A].
        tr (float): Rise time or characteristic time scale of the waveform [s].
    """
    Ipeak: float = 0
    tr: float = 0
    @classmethod
    def evolve(cls, t, y, args):
        """
        Compute the derivative of a prescribed sinusoidal current waveform.

        Notes:
            The returned `I` is the time derivative dI/dt implied by the waveform,
            not the current itself.

        Args:
            t (float): Current simulation time [s].
            y (objects.State): Current simulation state (used to read y.circ parameters).
            args (objects.args): Simulation configuration and physical parameters.

        Returns:
            sineCircuit: Circuit derivative object with field (I=dI/dt).
        """
        return cls(
            I = y.circ.Ipeak * np.cos(t * scipyc.pi / y.circ.tr / 2) * scipyc.pi / y.circ.tr / 2
        )

@dataclass
class sineSquaredCircuit(abstractCircuit):
    """
    Prescribed sin^2-style current drive model.

    This model imposes a current waveform proportional to sin^2(...).
    The `evolve` method returns the instantaneous time derivative of 
    the current.

    Attributes:
        Ipeak (float): Peak current amplitude [A].
        tr (float): Rise time or characteristic time scale of the waveform [s].
    """
    Ipeak: float = 0
    tr: float = 0
    @classmethod
    def evolve(cls, t, y, args):
        """
        Compute the derivative of a prescribed sin^2 current waveform.

        Notes:
            The returned `I` is the time derivative dI/dt implied by the waveform,
            not the current itself.

        Args:
            t (float): Current simulation time [s].
            y (objects.State): Current simulation state (used to read y.circ parameters).
            args (objects.args): Simulation configuration and physical parameters.

        Returns:
            sineSquaredCircuit: Circuit derivative object with field (I=dI/dt).
        """
        return cls(
            I = y.circ.Ipeak * np.cos(t * scipyc.pi / y.circ.tr / 2) * scipyc.pi / y.circ.tr *
            np.sin(scipyc.pi * t / y.circ.tr / 2)
        )

@dataclass
class RLCCircut(abstractCircuit):
    """
    Placeholder RLC circuit model.

    This class appears intended to represent an RLC circuit with capacitor voltage Vc,
    inductance L, and capacitance C, but the `evolve` method is currently unfinished
    (uses ellipsis for the current derivative).

    Attributes:
        I (float): Circuit current [A] (or dI/dt in derivative form).
        Vc (float): Capacitor voltage [V] (or dVc/dt in derivative form).
        L (float): Inductance [H].
        C (float): Capacitance [F].
    """
    Vc: float = 0
    L: float = 0
    C: float = 0
    @classmethod
    def evolve(cls, t, y, args):
        """
        Compute derivatives for an RLC circuit (unfinished implementation).

        Current behavior:
            - Computes Vc_dot = -I / C
            - Leaves I derivative unspecified (ellipsis)

        Args:
            t (float): Current simulation time [s].
            y (objects.State): Current simulation state.
            args (objects.args): Simulation configuration and physical parameters.

        Returns:
            RLCCircut: Circuit derivative object (incomplete; `I` derivative not implemented).
        """
        return cls(
            Vc = - y.circ.I / y.circ.C,
            I = ...
        )
