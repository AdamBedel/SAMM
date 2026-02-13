import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

#import tables
path = "./EOS_Tables/TempBeResistivity.csv"
temp_res = pd.read_csv(path)

T_disc = temp_res["T [K]"].values
eta_disc = temp_res["resistivity [10e-8ohm-m]"].values

#Convert units
eta_disc = eta_disc * 1e-8

def resistivity_Be(T):
    eta_interp = interp1d(
        T_disc,
        eta_disc,
        kind='cubic',
        bounds_error=False,
        fill_value="extrapolate"
    )
    if T > 1200:
        return eta_interp(T)
    return eta_interp(T)

