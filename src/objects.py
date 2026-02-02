# from __future__ import annotations
from dataclasses import dataclass, fields, is_dataclass
import numpy as np
from . import circuits, physics
from typing import Tuple, Any, Type
import scipy.constants as scipyc
from scipy.interpolate import interp1d

N_SHELLS = 120

CIRCUIT_CLASSES = {
    'sineCircuit': circuits.sineCircuit,
    'sineSquaredCircuit': circuits.sineSquaredCircuit,
    'voltageDrivenCircuit' : circuits.voltageDrivenCircuit,
    'LCCircuit' : circuits.LCCircuit
}

@dataclass
class ColdCurve:
    a1 : float
    a2 : float
    g1 : float
    g2 : float

@dataclass
class Material:
    a : float
    beta: float
    res : float
    rho: float
    cc : ColdCurve

    @classmethod
    def Beryllium(cls):
        return cls(
            a = 9.01222,
            beta = 3.683,
            res = 36e-9,
            rho = 1850,
            cc = ColdCurve(
                a1 = 130e9,
                a2 = 3.9993,
                g1 = 1.85,
                g2 = 1.18
            )
        )

@dataclass
class args:
    """Contains all static arguments to the ODE solver. """
    aq : float
    btheta : np.ndarray 
    Bz : float  # axial field [T]
    brems : bool  # if we should include brems
    circ_init: circuits.abstractCircuit
    f_tritium: float  # tritium fraction
    h: float  # liner height [m]
    mat: Material  # liner material
    n_shells: int  # number of liner shells
    ph_duration: float  # preheat duration [s]
    ph_energy: float  # preheat energy [J]
    ph_time: float  # preheat time [s]
    prefill_density: float  # gas prefill density [mg/cc] = [kg/m3]
    prefill_temperature: float  # gas prefill temperature [K]
    rg: float  # gas radius [m]
    rl0: float  # liner radius [m]
    rrc: float  # return can radius [m]

    @classmethod
    def default(cls):
        return cls(
            aq = 2,
            btheta = np.ones(N_SHELLS) * -1,
            Bz = 30,
            brems = True,
            circ_init = circuits.LCCircuit(
                I = 0,
                L = 16e-9,
                V = 6.79e6,
                C = 253e-9
            ),
            f_tritium = 0.5,
            h = 5.0e-3,
            mat = Material.Beryllium(),
            n_shells = N_SHELLS,
            ph_duration = 8e-9,
            ph_energy = 8000,  # 8 kJ
            ph_time = 60e-9,
            prefill_density = 3,
            prefill_temperature = 300,
            rg = 2.7e-3,
            rl0 = 3.24e-3,
            rrc = 10e-3,
        )

    # @classmethod
    # def default_HW3(cls):
    #     return cls(
    #         Bz = 10,
    #         circ_init = circuits.LCCircuit(
    #             I = 0,
    #             L = 16e-9,
    #             V = 5.03e6,
    #             C = 253e-9
    #         ),
    #         h = 7.5e-3,
    #         mat = Material.Beryllium(),
    #         rg = 2.325e-3,
    #         rl = 2.79e-3,
    #         rrc = 10e-3,
    #     )
    
    def flatten(self):
        return dataclass_to_tuple_with_type(self)

@dataclass
class State:
    """Contains the current state of the simulation. All of these variables
    have an evolution equation. """
    circ: circuits.abstractCircuit
    eg: float
    Nd: float
    Nt: float
    rg: float

    Ndt_neut: float = 0  # DT neutrons produced
    Ndd_neut : float = 0  # DD neutrons produced
    rl: np.ndarray = np.zeros(N_SHELLS)  # liner radiuses
    vl: np.ndarray = np.zeros(N_SHELLS)  # liner velocity
    vg: float = 0  # gas velocity

    def flatten(self):
        return toVector(self)

def toTuple(dataclass):
    flat = []

    for f in fields(dataclass):
        value = getattr(dataclass, f.name)
        if is_dataclass(value):
            flat.extend(toVector(value))
        else:
            flat.append(value)

    return tuple(flat)

@dataclass
class StateSeries:
    circ: np.ndarray
    eg: np.ndarray
    Nd: np.ndarray
    Nt: np.ndarray
    Ndd_neut : np.ndarray
    Ndt_neut : np.ndarray
    rg: np.ndarray
    rl: np.ndarray  # shape (N, N_SHELLS)
    vl: np.ndarray  # shape (N, N_SHELLS)
    vg: np.ndarray

def stack_states(states):
    """
    Convert an array of State objects (length N)
    into a StateSeries where each field is stacked
    over time.  rl and vl become 2D.
    """
    N = len(states)

    return StateSeries(
        circ=np.array([s.circ for s in states], dtype=object),
        eg=np.array([s.eg  for s in states]),
        Nd=np.array([s.Nd  for s in states]),
        Nt=np.array([s.Nt  for s in states]),
        Ndd_neut=np.array([s.Ndd_neut  for s in states]),
        Ndt_neut=np.array([s.Ndt_neut  for s in states]),
        rg=np.array([s.rg  for s in states]),
        rl=np.array([s.rl  for s in states]),   # shape (N, N_SHELLS)
        vl=np.array([s.vl  for s in states]),   # shape (N, N_SHELLS)
        vg=np.array([s.vg  for s in states]),
    )

def toVector(dataclass):
    flat = []

    for f in fields(dataclass):
        value = getattr(dataclass, f.name)

        if is_dataclass(value):
            flat.extend(toVector(value))

        # ---- NEW: if value is a numpy array, flatten it into scalars ----
        elif isinstance(value, np.ndarray):
            flat.extend(value.tolist())

        else:
            flat.append(value)

    return np.array(flat, dtype=float)

def vector_to_dataclass(arr: np.ndarray, cls: Any, circ_type: Type) -> Any:
    """
    Reconstruct a dataclass from a 1-D array OR a 2-D array (time × state).
    Any numpy array fields are assumed to have length N_SHELLS.
    """

    # If we were passed a matrix of many states, map over columns
    if arr.ndim == 2:
        return np.array([
            vector_to_dataclass(arr[:, i], cls, circ_type)
            for i in range(arr.shape[1])
        ])

    # Otherwise arr is a single state vector
    def _fill(cls, arr: np.ndarray, idx: int, circ_type: Type):
        kwargs = {}
        for f in fields(cls):

            f_type = f.type

            # --- nested dataclass (circuit or other) ---
            if is_dataclass(f_type):
                if issubclass(circ_type, f_type):
                    nested_obj, idx = _fill(circ_type, arr, idx, circ_type)
                else:
                    nested_obj, idx = _fill(f_type, arr, idx, circ_type)
                kwargs[f.name] = nested_obj
                continue

            # --- numpy array fields (rl, vl) ---
            if f_type is np.ndarray or f_type == np.ndarray:
                L = N_SHELLS   # ← use the constant
                kwargs[f.name] = np.array(arr[idx:idx+L], dtype=float)
                idx += L
                continue

            # --- scalar ---
            kwargs[f.name] = float(arr[idx])
            idx += 1

        return cls(**kwargs), idx

    obj, _ = _fill(cls, arr, 0, circ_type)
    return obj


def tuple_to_dataclass(data: Tuple[float, ...], cls: Any) -> Any:
    def _fill(cls: Any, data: Tuple[float, ...], idx: int) -> Tuple[Any, int]:
        kwargs = {}
        for f in fields(cls):
            field_type = f.type
            if is_dataclass(field_type):
                nested_obj, idx = _fill(field_type, data, idx)
                kwargs[f.name] = nested_obj
            else:
                kwargs[f.name] = data[idx]
                idx += 1
        return cls(**kwargs), idx

    obj, _ = _fill(cls, data, 0)
    return obj

def dataclass_to_tuple_with_type(obj: Any) -> Tuple[Any, ...]:
    """
    Flatten dataclass including type tag for subclasses of abstractCircuit.
    """
    if isinstance(obj, circuits.abstractCircuit):
        # Include class name as first element
        cls_name = obj.__class__.__name__
        flat_data = [cls_name]
        for f in fields(obj):
            val = getattr(obj, f.name)
            # For simple fields, just append values; for nested dataclasses recurse
            if is_dataclass(val):
                flat_data.extend(dataclass_to_tuple_with_type(val))
            else:
                flat_data.append(val)
        return tuple(flat_data)
    elif is_dataclass(obj):
        # For other dataclasses, no type tag needed
        flat_data = []
        for f in fields(obj):
            val = getattr(obj, f.name)
            if is_dataclass(val):
                flat_data.extend(dataclass_to_tuple_with_type(val))
            else:
                flat_data.append(val)
        return tuple(flat_data)
    else:
        raise TypeError(f"Unexpected type {type(obj)}")


def tuple_to_dataclass_with_type(data: Tuple[Any, ...], cls: Type) -> Any:
    def _fill(cls: Type, data: Tuple[Any, ...], idx: int) -> Tuple[Any, int]:
        kwargs = {}
        for f in fields(cls):
            f_type = f.type
            if isinstance(f_type, type) and issubclass(f_type, circuits.abstractCircuit):
                # Handle abstractCircuit subclass
                cls_name = data[idx]
                idx += 1
                concrete_cls = CIRCUIT_CLASSES.get(cls_name)
                if concrete_cls is None:
                    raise ValueError(f"Unknown circuit class '{cls_name}'")
                val, idx = _fill(concrete_cls, data, idx)
                kwargs[f.name] = val
            # elif is_dataclass(f_type):
            #     val, idx = _fill(f_type, data, idx)
            #     kwargs[f.name] = val
            elif is_dataclass(f.type):
                val, idx = _fill(f.type, data, idx)
                kwargs[f.name] = val
            else:
                kwargs[f.name] = data[idx]
                idx += 1
        
        return cls(**kwargs), idx

    obj, _ = _fill(cls, data, 0)
    return obj
