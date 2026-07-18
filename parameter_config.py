"""
parameter_config.py

Store all 3R2C parameters used in the MPC experiments.

There are three parameter sets:

1. TRUE_PARAMS
   The real building parameters used by the simulator.

2. GENERIC_PARAMS
   The average parameters computed from env_param.csv.
   These are used as the initial guess for parameter identification.

3. PARAMETER_BOUNDS
   Lower and upper bounds used during parameter identification.
"""

from copy import deepcopy


# ============================================================
# True parameters
# ============================================================

TRUE_PARAMS = {
    "C_env": 3.1996e6,
    "C_air": 3.5187e5,
    "R_rc": 0.00706,
    "R_oe": 0.02707,
    "R_er": 0.00369,
}


# ============================================================
# Generic parameters
# Mean values computed from env_param.csv (100 environments)
# ============================================================

GENERIC_PARAMS = {
    "C_env": 3.621417e6,
    "C_air": 3.953354e5,
    "R_rc": 0.007260759,
    "R_oe": 0.03000755,
    "R_er": 0.004009136,
}


# ============================================================
# Parameter bounds for identification
# (can be adjusted later if necessary)
# ============================================================

PARAMETER_BOUNDS = {
    "C_env": (1.0e6, 6.0e6),
    "C_air": (2.0e5, 6.0e5),
    "R_rc": (0.001, 0.020),
    "R_oe": (0.005, 0.050),
    "R_er": (0.0005, 0.010),
}


# ============================================================
# Helper functions
# ============================================================

def get_true_params():
    """Return a copy of the true parameters."""
    return deepcopy(TRUE_PARAMS)


def get_generic_params():
    """Return a copy of the generic parameters."""
    return deepcopy(GENERIC_PARAMS)


def get_parameter_bounds():
    """Return a copy of the parameter bounds."""
    return deepcopy(PARAMETER_BOUNDS)


def print_parameter_comparison():
    """Print the true and generic parameters."""

    print("=" * 65)
    print("3R2C Parameter Comparison")
    print("=" * 65)

    for key in TRUE_PARAMS:

        true_value = TRUE_PARAMS[key]
        generic_value = GENERIC_PARAMS[key]

        rel_error = abs(generic_value - true_value) / abs(true_value) * 100

        print(
            f"{key:<8}"
            f"True = {true_value:12.6g}    "
            f"Generic = {generic_value:12.6g}    "
            f"Difference = {rel_error:6.2f}%"
        )

    print("=" * 65)


if __name__ == "__main__":

    print_parameter_comparison()