from abc import ABC, abstractmethod
from dataclasses import dataclass
import numpy as np
import scipy.constants as scipyc
from scipy.interpolate import interp1d

@dataclass
class abstractCircuit(ABC):
    I: float  # current through load

    @classmethod
    @abstractmethod
    def evolve(cls, t, y, args):
        return cls(I = 0)

@dataclass
class voltageDrivenCircuit(abstractCircuit):
    L: float = 0
    @classmethod
    def evolve(cls, t, y, args):
        L_dot = - scipyc.mu_0 * args.h / (2 *scipyc.pi) * y.vl / y.rl[-1]
        I_dot = (cls.explicit_voltage(t) - L_dot * y.circ.I) / y.circ.L
        return cls(
            I = I_dot,
            L = L_dot
        )
    
    @staticmethod
    def explicit_voltage(t_eval) -> float:
        t = np.linspace(0, 250e-9, 100)
        V_array = 275e3 * np.cos(t * scipyc.pi / 200e-9)
        return interp1d(t, V_array)(t_eval)

@dataclass
class LCCircuit(abstractCircuit):
    L: float
    V: float
    C: float = 0
    @classmethod
    def evolve(cls, t, y, args):
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
    Ipeak: float = 0
    tr: float = 0
    @classmethod
    def evolve(cls, t, y, args):
        return cls(
            I = y.circ.Ipeak * np.cos(t * scipyc.pi / y.circ.tr / 2) * scipyc.pi / y.circ.tr / 2
        )

@dataclass
class sineSquaredCircuit(abstractCircuit):
    Ipeak: float = 0
    tr: float = 0
    @classmethod
    def evolve(cls, t, y, args):
        return cls(
            I = y.circ.Ipeak * np.cos(t * scipyc.pi / y.circ.tr / 2) * scipyc.pi / y.circ.tr *
            np.sin(scipyc.pi * t / y.circ.tr / 2)
        )

@dataclass
class RLCCircut(abstractCircuit):
    Vc: float = 0
    L: float = 0
    C: float = 0
    @classmethod
    def evolve(cls, t, y, args):
        return cls(
            Vc = - y.circ.I / y.circ.C,
            I = ...
        )