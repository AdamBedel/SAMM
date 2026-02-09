# from __future__ import annotations
from dataclasses import dataclass, fields, is_dataclass, field
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
    """
    Parameters for a cold-curve (EOS) pressure model.

    Attributes:
        a1 (float): Cold-curve coefficient.
        a2 (float): Cold-curve coefficient.
        g1 (float): Cold-curve exponent/coefficient.
        g2 (float): Cold-curve exponent/coefficient.
    """
    a1 : float
    a2 : float
    g1 : float
    g2 : float

@dataclass
class Material:
    """
    Liner material properties for the SAMM simulation.

    This dataclass groups together material constants used throughout the model,
    including resistivity, density, and cold-curve parameters.

    Attributes:
        a (float): Atomic mass or effective atomic weight [g/mol].
        beta (float): Exponent used in the assumed B_theta radial profile.
        res (float): Electrical resistivity [Ohm·m].
        rho (float): Mass density [kg/m^3].
        cc (ColdCurve): Cold-curve parameter set for compressive pressure modeling.
    """
    a : float
    beta: float
    res : float
    rho: float
    cc : ColdCurve

    @classmethod
    def Beryllium(cls):
        """
        Construct a Material instance with built-in beryllium parameters.

        Returns:
            Material: Material parameter set for Be, including its cold-curve coefficients.
        """
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
    """
    Static simulation configuration parameters for SAMM.

    This dataclass holds constants and runtime configuration used by the ODE system,
    including initial circuit configuration, geometry, material parameters, preheat,
    and flags controlling which physics terms are enabled.

    Attributes:
        aq (float): Artificial viscosity strength or tuning parameter (model-specific).
        btheta (numpy.ndarray): Azimuthal magnetic field profile.
        Bz (float): Applied axial magnetic field [T].
        brems (bool): Whether to include bremsstrahlung power loss.
        circ_init (circuits.abstractCircuit): Initial circuit object defining the drive.
        f_tritium (float): Tritium fraction in the fuel (0 to 1).
        h (float): Liner height [m].
        mat (Material): Liner material parameters.
        n_shells (int): Number of discretized liner shells.
        ph_duration (float): Preheat duration [s].
        ph_energy (float): Total preheat energy [J].
        ph_time (float): Time at which preheat is applied [s].
        prefill_density (float): Gas prefill density [mg/cc] (numerically equal to kg/m^3).
        prefill_temperature (float): Gas prefill temperature [K].
        rg (float): Initial gas radius [m].
        rl0 (float): Initial liner outer radius [m].
        rrc (float): Return can radius [m].
    """
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
        """
        Construct a default configuration for SAMM.

        Returns:
            args: Default parameter set used for typical runs.
        """
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
        """
        Flatten this args dataclass into a tuple representation.

        This uses `dataclass_to_tuple_with_type` so that circuit subclasses can be
        round-tripped by including a type tag (class name) for abstractCircuit objects.

        Returns:
            tuple[Any, ...]: Flattened tuple representation of args.
        """
        return dataclass_to_tuple_with_type(self)

@dataclass
class State:
    """Contains the current state of the simulation. All of these variables
    have an evolution equation. """
    """
    Dynamic simulation state variables for SAMM.

    These fields define the evolving ODE state: fuel energy and composition,
    radii/velocities, neutron bookkeeping, and the coupled circuit state.

    Notes:
        - `rl` and `vl` are arrays of length N_SHELLS.
        - `circ` is a circuit dataclass instance (subclass of abstractCircuit).
        - `flatten()` returns a numeric vector suitable for ODE integrators.

    Attributes:
        circ (circuits.abstractCircuit): Circuit state object.
        eg (float): Fuel internal energy [J].
        Nd (float): Deuterium particle count.
        Nt (float): Tritium particle count.
        rg (float): Fuel radius [m].
        Ndt_neut (float): Cumulative DT neutrons produced.
        Ndd_neut (float): Cumulative DD neutrons produced.
        rl (numpy.ndarray): Liner shell interface radii [m], shape (N_SHELLS,).
        vl (numpy.ndarray): Liner shell interface velocities [m/s], shape (N_SHELLS,).
        vg (float): Fuel radial velocity [m/s].
    """
    circ: circuits.abstractCircuit
    eg: float
    Nd: float
    Nt: float
    rg: float

    Ndt_neut: float = 0  # DT neutrons produced
    Ndd_neut : float = 0  # DD neutrons produced
    rl: np.ndarray = field(default_factory=lambda: np.zeros(N_SHELLS))
    vl: np.ndarray = field(default_factory=lambda: np.zeros(N_SHELLS))
    vg: float = 0  # gas velocity

    def flatten(self):
        """
        Flatten the State dataclass into a 1-D numeric vector.

        Returns:
            numpy.ndarray: Flattened state vector suitable for ODE integration.
        """
        return toVector(self)

def toTuple(dataclass):
    """
    Flatten a (possibly nested) dataclass into a Python tuple.

    This version treats nested dataclasses recursively and appends scalar fields
    directly.

    Warning:
        In the code as written, nested dataclasses call `toVector(value)` instead
        of `toTuple(value)`. If intentional, document why; if not, this is a bug.

    Args:
        dataclass (Any): Dataclass instance to flatten.

    Returns:
        tuple: Flattened tuple of values.
    """
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
    """
    Time-series container for a simulation run.

    Each field is the stacked value of the corresponding `State` field across time.
    Scalars become 1-D arrays of length N (timesteps). Array fields such as `rl` and
    `vl` become 2-D arrays of shape (N, N_SHELLS).

    Attributes:
        circ (numpy.ndarray): Array of circuit objects (dtype=object), length N.
        eg (numpy.ndarray): Fuel energy over time [J], shape (N,).
        Nd (numpy.ndarray): Deuterium count over time, shape (N,).
        Nt (numpy.ndarray): Tritium count over time, shape (N,).
        Ndd_neut (numpy.ndarray): Cumulative DD neutrons, shape (N,).
        Ndt_neut (numpy.ndarray): Cumulative DT neutrons, shape (N,).
        rg (numpy.ndarray): Fuel radius over time [m], shape (N,).
        rl (numpy.ndarray): Liner radii over time [m], shape (N, N_SHELLS).
        vl (numpy.ndarray): Liner velocities over time [m/s], shape (N, N_SHELLS).
        vg (numpy.ndarray): Fuel velocity over time [m/s], shape (N,).
    """
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
    """
    Stack a sequence of State objects into a StateSeries.

    Args:
        states (Sequence[State]): Iterable of State objects of length N.

    Returns:
        StateSeries: Time-series container with each field stacked over time.
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
    """
    Flatten a (possibly nested) dataclass into a 1-D numeric numpy array.

    Nested dataclasses are flattened recursively. Numpy array fields are expanded
    element-by-element into scalars (e.g., rl and vl of length N_SHELLS).

    Args:
        dataclass (Any): Dataclass instance to flatten.

    Returns:
        numpy.ndarray: 1-D float array containing the flattened values.
    """
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
    """
    Reconstruct a dataclass instance from a flattened numeric vector.

    Supports:
      - 1-D input representing a single state vector
      - 2-D input where each column is one state vector (state_dim × time)

    Conventions:
      - Nested dataclass fields are filled recursively.
      - Fields annotated as numpy.ndarray are assumed to have length N_SHELLS.
      - Circuit fields are reconstructed using `circ_type` so the correct circuit
        subclass is instantiated.

    Args:
        arr (numpy.ndarray): Flattened state vector (1-D) or stacked state matrix (2-D).
        cls (Any): Dataclass type to reconstruct (e.g., State).
        circ_type (Type): Concrete circuit class to use for the `circ` field.

    Returns:
        Any: Reconstructed dataclass instance (or numpy array of instances if arr is 2-D).
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
    """
    Reconstruct a dataclass instance from a flattened tuple.

    This is a tuple-based counterpart to `vector_to_dataclass`, but it does not
    handle numpy array expansion or circuit subclass tagging.

    Args:
        data (tuple[float, ...]): Flattened tuple representation.
        cls (Any): Dataclass type to reconstruct.

    Returns:
        Any: Reconstructed dataclass instance.
    """
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
    """
    Flatten a dataclass to a tuple, including a type tag for circuit subclasses.

    If `obj` is an instance of `circuits.abstractCircuit`, the first element of
    the tuple is the class name (e.g., "LCCircuit") so that deserialization can
    reconstruct the correct concrete circuit class.

    For non-circuit dataclasses, fields are flattened recursively without a type tag.

    Args:
        obj (Any): Dataclass instance to flatten.

    Returns:
        tuple[Any, ...]: Flattened tuple representation (with optional type tag).

    Raises:
        TypeError: If `obj` is not a dataclass instance or circuit instance.
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
    """
    Reconstruct a dataclass instance from a flattened tuple with circuit type tags.

    This is the inverse of `dataclass_to_tuple_with_type`. When encountering a field
    whose type is a subclass of `circuits.abstractCircuit`, the function expects the
    next element in the tuple to be a string class name (the type tag). It then uses
    `CIRCUIT_CLASSES` to map that name to a concrete circuit class for reconstruction.

    Args:
        data (tuple[Any, ...]): Flattened tuple representation, possibly including
            circuit type tags.
        cls (Type): Dataclass type to reconstruct (e.g., args or State).

    Returns:
        Any: Reconstructed dataclass instance.

    Raises:
        ValueError: If a circuit type tag is not recognized.
    """
    #why
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
