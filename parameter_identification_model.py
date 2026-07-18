"""
parameter_identification_model.py

Exact 3R2C parameter-identification model.

This module reproduces the thermal and PI-control transition used by
Env_develop_mpc_draft.py.

The model does NOT use an externally estimated HVAC heat input. Instead,
for every candidate parameter set, it reconstructs the six internal
five-minute PI-controller updates:

    1. Reset PI integral when the ZAT setpoint changes sufficiently.
    2. Compute the PI damper command.
    3. Compute supply airflow.
    4. Compute supply-air heat transfer.
    5. Compute terminal reheat.
    6. Apply the exact discrete 3R2C state-space transition.

The main public functions are:

    load_identification_dataset(...)
    simulate_one_step_predictions(...)
    simulate_rollout(...)
    calculate_simulation_metrics(...)
    compute_residual_vector(...)
    compute_loss(...)

They preserve the interface expected by parameter_identification.py.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from scipy import signal


# ============================================================
# Parameter definitions
# ============================================================

PARAMETER_NAMES: Tuple[str, ...] = (
    "C_env",
    "C_air",
    "R_rc",
    "R_oe",
    "R_er",
)

ParameterInput = Union[
    "ThreeRTwoCParameters",
    Mapping[str, float],
    Sequence[float],
    np.ndarray,
]


@dataclass(frozen=True)
class ThreeRTwoCParameters:
    """Physical parameters of the two-state 3R2C building model."""

    C_env: float
    C_air: float
    R_rc: float
    R_oe: float
    R_er: float

    def __post_init__(self) -> None:
        values = self.to_array()

        if not np.all(np.isfinite(values)):
            raise ValueError(
                "All 3R2C parameters must be finite."
            )

        if np.any(values <= 0.0):
            raise ValueError(
                "All 3R2C parameters must be positive."
            )

    def to_array(self) -> np.ndarray:
        """Return parameters in PARAMETER_NAMES order."""

        return np.array(
            [
                self.C_env,
                self.C_air,
                self.R_rc,
                self.R_oe,
                self.R_er,
            ],
            dtype=np.float64,
        )

    def to_dict(self) -> Dict[str, float]:
        """Return parameters as a standard dictionary."""

        return {
            "C_env": float(self.C_env),
            "C_air": float(self.C_air),
            "R_rc": float(self.R_rc),
            "R_oe": float(self.R_oe),
            "R_er": float(self.R_er),
        }

    @classmethod
    def from_array(
        cls,
        values: Sequence[float],
    ) -> "ThreeRTwoCParameters":
        """Create parameters from a five-element vector."""

        array = np.asarray(
            values,
            dtype=np.float64,
        ).reshape(-1)

        if array.size != len(PARAMETER_NAMES):
            raise ValueError(
                f"Expected {len(PARAMETER_NAMES)} parameter values, "
                f"but received {array.size}."
            )

        return cls(
            C_env=float(array[0]),
            C_air=float(array[1]),
            R_rc=float(array[2]),
            R_oe=float(array[3]),
            R_er=float(array[4]),
        )

    @classmethod
    def from_mapping(
        cls,
        values: Mapping[str, float],
    ) -> "ThreeRTwoCParameters":
        """Create parameters from a mapping."""

        missing = [
            name
            for name in PARAMETER_NAMES
            if name not in values
        ]

        if missing:
            raise KeyError(
                f"Missing parameter values: {missing}"
            )

        return cls(
            C_env=float(values["C_env"]),
            C_air=float(values["C_air"]),
            R_rc=float(values["R_rc"]),
            R_oe=float(values["R_oe"]),
            R_er=float(values["R_er"]),
        )


def convert_parameters(
    parameters: ParameterInput,
) -> ThreeRTwoCParameters:
    """Convert any supported parameter representation."""

    if isinstance(
        parameters,
        ThreeRTwoCParameters,
    ):
        return parameters

    if isinstance(parameters, Mapping):
        return ThreeRTwoCParameters.from_mapping(
            parameters
        )

    return ThreeRTwoCParameters.from_array(
        parameters
    )


def parameter_vector_to_dict(
    values: Sequence[float],
) -> Dict[str, float]:
    """Convert a parameter vector to a dictionary."""

    return ThreeRTwoCParameters.from_array(
        values
    ).to_dict()


def parameter_dict_to_vector(
    values: Mapping[str, float],
) -> np.ndarray:
    """Convert a parameter dictionary to a vector."""

    return ThreeRTwoCParameters.from_mapping(
        values
    ).to_array()


# ============================================================
# Environment constants
# ============================================================

SOLAR_GAIN_TO_ENVELOPE = 0.3

PI_INTERVAL_SECONDS = 300.0
DEFAULT_OUTER_DT_SECONDS = 1800.0

PI_KP = 15.0
PI_KI = 0.02
PI_SETPOINT_RESET_THRESHOLD = 0.5

CP_AIR = 1004.0

M_DOT_MIN = 0.080939
M_DOT_MAX = M_DOT_MIN * 550.0 / 140.0

CAPACITY_SCALE = 1.0 / 3.0

Q_REHEAT_MAX_W = 300.0

DAMPER_MIN_PERCENT = 0.0
DAMPER_MAX_PERCENT = 100.0

REHEAT_MIN_FLOW_THRESHOLD_PERCENT = 1.0


# ============================================================
# Dataset definitions
# ============================================================

REQUIRED_DATASET_COLUMNS: Tuple[str, ...] = (
    "T_env_start",
    "T_zone_start",

    "SAT_sp",
    "ZAT_sp",

    "integral_error_start",
    "prev_zat_sp_start",
    "damper_signal_prev_start",

    "T_cor_used",
    "T_out_used",
    "Qsg_used",
    "Qint_used",

    "T_env_end",
    "T_zone_end",
)

OPTIONAL_DATASET_COLUMNS: Tuple[str, ...] = (
    "row_index",
    "time_hour_start",
    "time_hour_end",
    "dt_seconds",

    "integral_error_end",
    "prev_zat_sp_end",
    "damper_signal_end",
    "m_fan_end",
)


def load_identification_dataset(
    file_path: Union[str, Path],
) -> pd.DataFrame:
    """
    Load and validate the new identification dataset.

    The dataset must have been generated by the updated
    collect_identification_data.py.
    """

    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(
            f"Identification dataset not found: {file_path}"
        )

    dataset = pd.read_csv(file_path)

    missing_columns = [
        column
        for column in REQUIRED_DATASET_COLUMNS
        if column not in dataset.columns
    ]

    if missing_columns:
        raise ValueError(
            "The dataset does not contain the required exact-transition "
            f"columns: {missing_columns}\n"
            "Regenerate the dataset using the updated "
            "collect_identification_data.py."
        )

    if dataset.empty:
        raise ValueError(
            "The identification dataset is empty."
        )

    numeric_columns = list(
        REQUIRED_DATASET_COLUMNS
    )

    for column in OPTIONAL_DATASET_COLUMNS:
        if column in dataset.columns:
            numeric_columns.append(column)

    for column in numeric_columns:
        dataset[column] = pd.to_numeric(
            dataset[column],
            errors="coerce",
        )

    # prev_zat_sp_start is intentionally allowed to be NaN in the first row.
    required_finite_columns = [
        column
        for column in REQUIRED_DATASET_COLUMNS
        if column != "prev_zat_sp_start"
    ]

    invalid_columns = [
        column
        for column in required_finite_columns
        if not np.all(
            np.isfinite(
                dataset[column].to_numpy(
                    dtype=np.float64
                )
            )
        )
    ]

    if invalid_columns:
        raise ValueError(
            "The following required columns contain missing or "
            f"non-finite values: {invalid_columns}"
        )

    if "row_index" not in dataset.columns:
        dataset.insert(
            0,
            "row_index",
            np.arange(
                len(dataset),
                dtype=np.int64,
            ),
        )

    dataset = dataset.reset_index(
        drop=True
    )

    return dataset


# ============================================================
# Continuous and discrete 3R2C models
# ============================================================

def build_continuous_matrices(
    parameters: ParameterInput,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build continuous-time A and B matrices.

    Input order is:

        [T_cor, T_out, Qsg, Qint, Q_HVAC]

    State order is:

        [T_env, T_zone]
    """

    p = convert_parameters(parameters)

    A = np.zeros(
        (2, 2),
        dtype=np.float64,
    )

    B = np.zeros(
        (2, 5),
        dtype=np.float64,
    )

    A[0, 0] = (
        -1.0
        / p.C_env
        * (
            1.0 / p.R_er
            + 1.0 / p.R_oe
        )
    )

    A[0, 1] = (
        1.0
        / (
            p.C_env
            * p.R_er
        )
    )

    A[1, 0] = (
        1.0
        / (
            p.C_air
            * p.R_er
        )
    )

    A[1, 1] = (
        -1.0
        / p.C_air
        * (
            1.0 / p.R_er
            + 1.0 / p.R_rc
        )
    )

    # T_cor has no direct effect on the envelope.
    B[0, 0] = 0.0

    # Outdoor temperature affects the envelope through R_oe.
    B[0, 1] = (
        1.0
        / (
            p.C_env
            * p.R_oe
        )
    )

    # 30% of solar gain enters the envelope.
    B[0, 2] = (
        SOLAR_GAIN_TO_ENVELOPE
        / p.C_env
    )

    B[0, 3] = 0.0
    B[0, 4] = 0.0

    # Corridor temperature affects the zone through R_rc.
    B[1, 0] = (
        1.0
        / (
            p.C_air
            * p.R_rc
        )
    )

    B[1, 1] = 0.0

    # 70% of solar gain enters the zone.
    B[1, 2] = (
        1.0
        - SOLAR_GAIN_TO_ENVELOPE
    ) / p.C_air

    # Internal heat and HVAC heat both enter the zone.
    B[1, 3] = 1.0 / p.C_air
    B[1, 4] = 1.0 / p.C_air

    return A, B


def discretize_state_space(
    A: np.ndarray,
    B: np.ndarray,
    dt_seconds: float = PI_INTERVAL_SECONDS,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply the same zero-order-hold discretization as the environment.
    """

    dt_seconds = float(dt_seconds)

    if not np.isfinite(dt_seconds) or dt_seconds <= 0.0:
        raise ValueError(
            "dt_seconds must be a positive finite value."
        )

    discrete_system = signal.StateSpace(
        A,
        B,
        np.array(
            [[1.0, 0.0]],
            dtype=np.float64,
        ),
        np.zeros(
            5,
            dtype=np.float64,
        ),
    ).to_discrete(
        dt=dt_seconds
    )

    return (
        np.asarray(
            discrete_system.A,
            dtype=np.float64,
        ),
        np.asarray(
            discrete_system.B,
            dtype=np.float64,
        ),
    )


def build_pi_discrete_matrices(
    parameters: ParameterInput,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build the exact 300-second matrices used in the PI loop."""

    A, B = build_continuous_matrices(
        parameters
    )

    return discretize_state_space(
        A=A,
        B=B,
        dt_seconds=PI_INTERVAL_SECONDS,
    )


# ============================================================
# Controller-state utilities
# ============================================================

@dataclass
class ControllerState:
    """PI controller state carried across outer transitions."""

    integral_error: float
    prev_zat_sp: Optional[float]
    damper_signal: float

    def copy(self) -> "ControllerState":
        return ControllerState(
            integral_error=float(
                self.integral_error
            ),
            prev_zat_sp=(
                None
                if self.prev_zat_sp is None
                else float(self.prev_zat_sp)
            ),
            damper_signal=float(
                self.damper_signal
            ),
        )


@dataclass
class TransitionResult:
    """Output from one reconstructed outer transition."""

    T_env_end: float
    T_zone_end: float

    integral_error_end: float
    prev_zat_sp_end: float
    damper_signal_end: float
    m_fan_end: float

    Q_air_last_W: float
    Q_reheat_last_W: float
    Q_hvac_last_W: float

    n_pi_loops: int


def _optional_float(
    value: Any,
) -> Optional[float]:
    """Interpret NaN/None as None."""

    if value is None:
        return None

    converted = float(value)

    if not np.isfinite(converted):
        return None

    return converted


def _get_outer_dt_seconds(
    row: Mapping[str, Any],
) -> float:
    """Read the outer step duration from a row."""

    if "dt_seconds" not in row:
        return DEFAULT_OUTER_DT_SECONDS

    dt_seconds = float(row["dt_seconds"])

    if not np.isfinite(dt_seconds) or dt_seconds <= 0.0:
        raise ValueError(
            f"Invalid dt_seconds value: {dt_seconds}"
        )

    return dt_seconds


def _calculate_number_of_pi_loops(
    outer_dt_seconds: float,
) -> int:
    """
    Match the environment's int(dt // pi_interval) behavior.
    """

    n_pi_loops = int(
        float(outer_dt_seconds)
        // PI_INTERVAL_SECONDS
    )

    if n_pi_loops < 1:
        n_pi_loops = 1

    return n_pi_loops


# ============================================================
# Exact outer-transition reconstruction
# ============================================================

def simulate_exact_transition(
    parameters: ParameterInput,
    T_env_start: float,
    T_zone_start: float,
    SAT_sp: float,
    ZAT_sp: float,
    T_cor: float,
    T_out: float,
    Qsg: float,
    Qint: float,
    controller_state: ControllerState,
    outer_dt_seconds: float = DEFAULT_OUTER_DT_SECONDS,
    discrete_matrices: Optional[
        Tuple[np.ndarray, np.ndarray]
    ] = None,
) -> TransitionResult:
    """
    Reproduce one complete outer environment transition.

    The operation ordering intentionally matches the environment code.
    """

    values_to_check = np.array(
        [
            T_env_start,
            T_zone_start,
            SAT_sp,
            ZAT_sp,
            T_cor,
            T_out,
            Qsg,
            Qint,
            controller_state.integral_error,
            controller_state.damper_signal,
            outer_dt_seconds,
        ],
        dtype=np.float64,
    )

    if not np.all(np.isfinite(values_to_check)):
        raise ValueError(
            "Transition inputs must contain only finite values."
        )

    if discrete_matrices is None:
        A_pi, B_pi = (
            build_pi_discrete_matrices(
                parameters
            )
        )
    else:
        A_pi, B_pi = discrete_matrices

    x_room = np.array(
        [
            float(T_env_start),
            float(T_zone_start),
        ],
        dtype=np.float64,
    )

    T_zone = float(T_zone_start)

    next_integral_error = float(
        controller_state.integral_error
    )

    previous_zat_sp = (
        controller_state.prev_zat_sp
    )

    if (
        previous_zat_sp is None
        or abs(
            float(ZAT_sp)
            - float(previous_zat_sp)
        )
        > PI_SETPOINT_RESET_THRESHOLD
    ):
        next_integral_error = 0.0

    next_prev_zat_sp = float(
        ZAT_sp
    )

    damper_signal = float(
        controller_state.damper_signal
    )

    n_pi_loops = (
        _calculate_number_of_pi_loops(
            outer_dt_seconds
        )
    )

    u_base = np.array(
        [
            float(T_cor),
            float(T_out),
            float(Qsg),
            float(Qint),
        ],
        dtype=np.float64,
    )

    m_fan_last = M_DOT_MIN

    Q_air_last = 0.0
    Q_reheat_last = 0.0
    Q_hvac_last = 0.0

    for _ in range(n_pi_loops):
        # This first calculation is intentionally retained because it is
        # present in the environment, although m_fan is recalculated after
        # the new damper command.
        m_fan = (
            M_DOT_MIN
            + (
                damper_signal
                / 100.0
            )
            * (
                M_DOT_MAX
                - M_DOT_MIN
            )
        )

        error = (
            T_zone
            - float(ZAT_sp)
        )

        raw_damper_command = (
            PI_KP * error
            + PI_KI
            * next_integral_error
        )

        damper_signal = float(
            np.clip(
                raw_damper_command,
                DAMPER_MIN_PERCENT,
                DAMPER_MAX_PERCENT,
            )
        )

        # Exact anti-windup condition used by the environment.
        if not (
            (
                raw_damper_command >= 99.9
                and error > 0.0
            )
            or (
                raw_damper_command <= 0.1
                and error < 0.0
            )
        ):
            next_integral_error += (
                error
                * PI_INTERVAL_SECONDS
            )

        m_fan = (
            M_DOT_MIN
            + (
                damper_signal
                / 100.0
            )
            * (
                M_DOT_MAX
                - M_DOT_MIN
            )
        )

        m_fan_last = float(m_fan)

        Q_air = (
            CAPACITY_SCALE
            * m_fan
            * CP_AIR
            * (
                float(SAT_sp)
                - T_zone
            )
        )

        at_min_flow = (
            damper_signal
            <= REHEAT_MIN_FLOW_THRESHOLD_PERCENT
        )

        if (
            at_min_flow
            and T_zone < float(ZAT_sp)
        ):
            reheat_signal = float(
                np.clip(
                    (
                        float(ZAT_sp)
                        - T_zone
                    )
                    / 3.0,
                    0.0,
                    1.0,
                )
            )

            Q_reheat = (
                reheat_signal
                * Q_REHEAT_MAX_W
            )
        else:
            Q_reheat = 0.0

        Q_hvac = (
            Q_air
            + Q_reheat
        )

        u_model = np.array(
            [
                u_base[0],
                u_base[1],
                u_base[2],
                u_base[3],
                Q_hvac,
            ],
            dtype=np.float64,
        )

        x_room = (
            A_pi @ x_room
            + B_pi @ u_model
        )

        T_zone = float(
            x_room[1]
        )

        Q_air_last = float(
            Q_air
        )

        Q_reheat_last = float(
            Q_reheat
        )

        Q_hvac_last = float(
            Q_hvac
        )

    return TransitionResult(
        T_env_end=float(
            x_room[0]
        ),
        T_zone_end=float(
            x_room[1]
        ),

        integral_error_end=float(
            next_integral_error
        ),
        prev_zat_sp_end=float(
            next_prev_zat_sp
        ),
        damper_signal_end=float(
            damper_signal
        ),
        m_fan_end=float(
            m_fan_last
        ),

        Q_air_last_W=float(
            Q_air_last
        ),
        Q_reheat_last_W=float(
            Q_reheat_last
        ),
        Q_hvac_last_W=float(
            Q_hvac_last
        ),

        n_pi_loops=int(
            n_pi_loops
        ),
    )


def predict_transition_from_row(
    parameters: ParameterInput,
    row: Mapping[str, Any],
    override_initial_thermal_state: Optional[
        Tuple[float, float]
    ] = None,
    override_controller_state: Optional[
        ControllerState
    ] = None,
    discrete_matrices: Optional[
        Tuple[np.ndarray, np.ndarray]
    ] = None,
) -> TransitionResult:
    """
    Predict one row using its stored inputs.

    override_* arguments are used by rollout simulation.
    """

    if override_initial_thermal_state is None:
        T_env_start = float(
            row["T_env_start"]
        )

        T_zone_start = float(
            row["T_zone_start"]
        )
    else:
        (
            T_env_start,
            T_zone_start,
        ) = override_initial_thermal_state

    if override_controller_state is None:
        controller_state = ControllerState(
            integral_error=float(
                row["integral_error_start"]
            ),
            prev_zat_sp=_optional_float(
                row["prev_zat_sp_start"]
            ),
            damper_signal=float(
                row[
                    "damper_signal_prev_start"
                ]
            ),
        )
    else:
        controller_state = (
            override_controller_state.copy()
        )

    outer_dt_seconds = (
        _get_outer_dt_seconds(row)
    )

    return simulate_exact_transition(
        parameters=parameters,

        T_env_start=T_env_start,
        T_zone_start=T_zone_start,

        SAT_sp=float(row["SAT_sp"]),
        ZAT_sp=float(row["ZAT_sp"]),

        T_cor=float(row["T_cor_used"]),
        T_out=float(row["T_out_used"]),
        Qsg=float(row["Qsg_used"]),
        Qint=float(row["Qint_used"]),

        controller_state=controller_state,

        outer_dt_seconds=(
            outer_dt_seconds
        ),

        discrete_matrices=(
            discrete_matrices
        ),
    )


# ============================================================
# Prediction table creation
# ============================================================

def _make_prediction_row(
    row_index: int,
    source_row: Mapping[str, Any],
    predicted_T_env: float,
    predicted_T_zone: float,
    transition_result: TransitionResult,
    simulation_type: str,
    T_env_model_start: float,
    T_zone_model_start: float,
) -> Dict[str, Any]:
    """Create a standard prediction output row."""

    measured_T_env = float(
        source_row["T_env_end"]
    )

    measured_T_zone = float(
        source_row["T_zone_end"]
    )

    T_env_error = (
        float(predicted_T_env)
        - measured_T_env
    )

    T_zone_error = (
        float(predicted_T_zone)
        - measured_T_zone
    )

    return {
        "row_index": int(
            source_row.get(
                "row_index",
                row_index,
            )
        ),
        "simulation_type":
            simulation_type,

        "T_env_start_measured": float(
            source_row["T_env_start"]
        ),
        "T_zone_start_measured": float(
            source_row["T_zone_start"]
        ),

        "T_env_start_model": float(
            T_env_model_start
        ),
        "T_zone_start_model": float(
            T_zone_model_start
        ),

        "T_env_measured":
            measured_T_env,
        "T_env_predicted": float(
            predicted_T_env
        ),
        "T_env_error": float(
            T_env_error
        ),

        "T_zone_measured":
            measured_T_zone,
        "T_zone_predicted": float(
            predicted_T_zone
        ),
        "T_zone_error": float(
            T_zone_error
        ),

        "SAT_sp": float(
            source_row["SAT_sp"]
        ),
        "ZAT_sp": float(
            source_row["ZAT_sp"]
        ),

        "T_cor_used": float(
            source_row["T_cor_used"]
        ),
        "T_out_used": float(
            source_row["T_out_used"]
        ),
        "Qsg_used": float(
            source_row["Qsg_used"]
        ),
        "Qint_used": float(
            source_row["Qint_used"]
        ),

        "integral_error_predicted_end":
            float(
                transition_result
                .integral_error_end
            ),

        "prev_zat_sp_predicted_end":
            float(
                transition_result
                .prev_zat_sp_end
            ),

        "damper_signal_predicted_end":
            float(
                transition_result
                .damper_signal_end
            ),

        "m_fan_predicted_end": float(
            transition_result
            .m_fan_end
        ),

        "Q_air_last_predicted_W":
            float(
                transition_result
                .Q_air_last_W
            ),

        "Q_reheat_last_predicted_W":
            float(
                transition_result
                .Q_reheat_last_W
            ),

        "Q_hvac_last_predicted_W":
            float(
                transition_result
                .Q_hvac_last_W
            ),

        "n_pi_loops": int(
            transition_result
            .n_pi_loops
        ),
    }


def simulate_one_step_predictions(
    parameters: ParameterInput,
    dataset: pd.DataFrame,
    integration_substeps: Optional[int] = None,
    method: Optional[str] = None,
    **_: Any,
) -> pd.DataFrame:
    """
    Simulate independent one-step predictions.

    Every prediction starts from the measured thermal state and stored
    controller state for that row.

    integration_substeps and method are accepted only for backward
    compatibility. They are ignored because this model always uses the
    environment's exact 300-second discrete transition.
    """

    del integration_substeps
    del method

    p = convert_parameters(parameters)

    A_pi, B_pi = (
        build_pi_discrete_matrices(p)
    )

    prediction_rows = []

    for local_index, row in dataset.iterrows():
        transition = (
            predict_transition_from_row(
                parameters=p,
                row=row,
                discrete_matrices=(
                    A_pi,
                    B_pi,
                ),
            )
        )

        prediction_rows.append(
            _make_prediction_row(
                row_index=int(local_index),
                source_row=row,
                predicted_T_env=(
                    transition.T_env_end
                ),
                predicted_T_zone=(
                    transition.T_zone_end
                ),
                transition_result=(
                    transition
                ),
                simulation_type=(
                    "one_step"
                ),
                T_env_model_start=float(
                    row["T_env_start"]
                ),
                T_zone_model_start=float(
                    row["T_zone_start"]
                ),
            )
        )

    return pd.DataFrame(
        prediction_rows
    )


def simulate_rollout(
    parameters: ParameterInput,
    dataset: pd.DataFrame,
    integration_substeps: Optional[int] = None,
    method: Optional[str] = None,
    reset_on_discontinuity: bool = True,
    continuity_tolerance: float = 1.0e-5,
    **_: Any,
) -> pd.DataFrame:
    """
    Simulate a sequential rollout over the dataset.

    Thermal and PI states are propagated from one row to the next.

    When reset_on_discontinuity=True, the rollout restarts from the
    measured state when the dataset contains a time discontinuity or
    appears to begin a new episode.
    """

    del integration_substeps
    del method

    if dataset.empty:
        raise ValueError(
            "Cannot simulate an empty dataset."
        )

    p = convert_parameters(parameters)

    A_pi, B_pi = (
        build_pi_discrete_matrices(p)
    )

    first_row = dataset.iloc[0]

    current_T_env = float(
        first_row["T_env_start"]
    )

    current_T_zone = float(
        first_row["T_zone_start"]
    )

    controller_state = ControllerState(
        integral_error=float(
            first_row[
                "integral_error_start"
            ]
        ),
        prev_zat_sp=_optional_float(
            first_row[
                "prev_zat_sp_start"
            ]
        ),
        damper_signal=float(
            first_row[
                "damper_signal_prev_start"
            ]
        ),
    )

    prediction_rows = []

    previous_source_row = None

    for local_index, row in dataset.iterrows():
        should_reset = False

        if (
            reset_on_discontinuity
            and previous_source_row is not None
        ):
            measured_env_gap = abs(
                float(row["T_env_start"])
                - float(
                    previous_source_row[
                        "T_env_end"
                    ]
                )
            )

            measured_zone_gap = abs(
                float(row["T_zone_start"])
                - float(
                    previous_source_row[
                        "T_zone_end"
                    ]
                )
            )

            time_gap = 0.0

            if (
                "time_hour_start"
                in dataset.columns
                and "time_hour_end"
                in dataset.columns
            ):
                time_gap = abs(
                    float(
                        row[
                            "time_hour_start"
                        ]
                    )
                    - float(
                        previous_source_row[
                            "time_hour_end"
                        ]
                    )
                )

            should_reset = (
                measured_env_gap
                > continuity_tolerance
                or measured_zone_gap
                > continuity_tolerance
                or time_gap
                > continuity_tolerance
            )

        if should_reset:
            current_T_env = float(
                row["T_env_start"]
            )

            current_T_zone = float(
                row["T_zone_start"]
            )

            controller_state = (
                ControllerState(
                    integral_error=float(
                        row[
                            "integral_error_start"
                        ]
                    ),
                    prev_zat_sp=(
                        _optional_float(
                            row[
                                "prev_zat_sp_start"
                            ]
                        )
                    ),
                    damper_signal=float(
                        row[
                            "damper_signal_prev_start"
                        ]
                    ),
                )
            )

        model_start_env = float(
            current_T_env
        )

        model_start_zone = float(
            current_T_zone
        )

        transition = (
            predict_transition_from_row(
                parameters=p,
                row=row,
                override_initial_thermal_state=(
                    current_T_env,
                    current_T_zone,
                ),
                override_controller_state=(
                    controller_state
                ),
                discrete_matrices=(
                    A_pi,
                    B_pi,
                ),
            )
        )

        prediction_rows.append(
            _make_prediction_row(
                row_index=int(local_index),
                source_row=row,
                predicted_T_env=(
                    transition.T_env_end
                ),
                predicted_T_zone=(
                    transition.T_zone_end
                ),
                transition_result=(
                    transition
                ),
                simulation_type=(
                    "rollout"
                ),
                T_env_model_start=(
                    model_start_env
                ),
                T_zone_model_start=(
                    model_start_zone
                ),
            )
        )

        current_T_env = float(
            transition.T_env_end
        )

        current_T_zone = float(
            transition.T_zone_end
        )

        controller_state = (
            ControllerState(
                integral_error=float(
                    transition
                    .integral_error_end
                ),
                prev_zat_sp=float(
                    transition
                    .prev_zat_sp_end
                ),
                damper_signal=float(
                    transition
                    .damper_signal_end
                ),
            )
        )

        previous_source_row = row

    return pd.DataFrame(
        prediction_rows
    )


# ============================================================
# Metrics
# ============================================================

@dataclass(frozen=True)
class SimulationMetrics:
    """Prediction error metrics."""

    envelope_rmse: float
    zone_rmse: float
    combined_rmse: float

    envelope_mae: float
    zone_mae: float
    combined_mae: float

    envelope_bias: float
    zone_bias: float

    envelope_max_abs_error: float
    zone_max_abs_error: float

    number_of_samples: int

    def to_dict(self) -> Dict[str, float]:
        """Convert metrics to a dictionary."""

        return {
            "envelope_rmse":
                float(self.envelope_rmse),
            "zone_rmse":
                float(self.zone_rmse),
            "combined_rmse":
                float(self.combined_rmse),

            "envelope_mae":
                float(self.envelope_mae),
            "zone_mae":
                float(self.zone_mae),
            "combined_mae":
                float(self.combined_mae),

            "envelope_bias":
                float(self.envelope_bias),
            "zone_bias":
                float(self.zone_bias),

            "envelope_max_abs_error":
                float(
                    self.envelope_max_abs_error
                ),
            "zone_max_abs_error":
                float(
                    self.zone_max_abs_error
                ),

            "number_of_samples":
                int(self.number_of_samples),
        }


def calculate_simulation_metrics(
    predictions: pd.DataFrame,
) -> SimulationMetrics:
    """Calculate standard prediction metrics."""

    required_columns = {
        "T_env_error",
        "T_zone_error",
    }

    missing = (
        required_columns
        - set(predictions.columns)
    )

    if missing:
        raise ValueError(
            "Prediction dataframe is missing "
            f"error columns: {sorted(missing)}"
        )

    if predictions.empty:
        raise ValueError(
            "Cannot calculate metrics from an empty dataframe."
        )

    envelope_error = predictions[
        "T_env_error"
    ].to_numpy(
        dtype=np.float64
    )

    zone_error = predictions[
        "T_zone_error"
    ].to_numpy(
        dtype=np.float64
    )

    if not (
        np.all(np.isfinite(envelope_error))
        and np.all(np.isfinite(zone_error))
    ):
        raise ValueError(
            "Prediction errors contain non-finite values."
        )

    envelope_rmse = float(
        np.sqrt(
            np.mean(
                envelope_error ** 2
            )
        )
    )

    zone_rmse = float(
        np.sqrt(
            np.mean(
                zone_error ** 2
            )
        )
    )

    combined_rmse = float(
        np.sqrt(
            np.mean(
                np.concatenate(
                    [
                        envelope_error,
                        zone_error,
                    ]
                )
                ** 2
            )
        )
    )

    envelope_mae = float(
        np.mean(
            np.abs(envelope_error)
        )
    )

    zone_mae = float(
        np.mean(
            np.abs(zone_error)
        )
    )

    combined_mae = float(
        np.mean(
            np.abs(
                np.concatenate(
                    [
                        envelope_error,
                        zone_error,
                    ]
                )
            )
        )
    )

    return SimulationMetrics(
        envelope_rmse=envelope_rmse,
        zone_rmse=zone_rmse,
        combined_rmse=combined_rmse,

        envelope_mae=envelope_mae,
        zone_mae=zone_mae,
        combined_mae=combined_mae,

        envelope_bias=float(
            np.mean(envelope_error)
        ),
        zone_bias=float(
            np.mean(zone_error)
        ),

        envelope_max_abs_error=float(
            np.max(
                np.abs(envelope_error)
            )
        ),
        zone_max_abs_error=float(
            np.max(
                np.abs(zone_error)
            )
        ),

        number_of_samples=int(
            len(predictions)
        ),
    )


# ============================================================
# Optimization residuals and losses
# ============================================================

def compute_residual_vector(
    parameters: ParameterInput,
    dataset: pd.DataFrame,
    envelope_weight: float = 1.0,
    zone_weight: float = 1.0,
    simulation_mode: str = "one_step",
    normalize_by_samples: bool = False,
    integration_substeps: Optional[int] = None,
    method: Optional[str] = None,
    **_: Any,
) -> np.ndarray:
    """
    Return the weighted residual vector used by least_squares.

    Residual order:

        [all envelope errors, all zone errors]

    Default weights are equal. To emphasize zone-temperature fitting,
    set zone_weight > envelope_weight.
    """

    del integration_substeps
    del method

    envelope_weight = float(
        envelope_weight
    )

    zone_weight = float(
        zone_weight
    )

    if (
        not np.isfinite(envelope_weight)
        or envelope_weight <= 0.0
    ):
        raise ValueError(
            "envelope_weight must be positive and finite."
        )

    if (
        not np.isfinite(zone_weight)
        or zone_weight <= 0.0
    ):
        raise ValueError(
            "zone_weight must be positive and finite."
        )

    normalized_mode = (
        simulation_mode
        .strip()
        .lower()
        .replace("-", "_")
    )

    if normalized_mode in {
        "one_step",
        "onestep",
        "one",
    }:
        predictions = (
            simulate_one_step_predictions(
                parameters=parameters,
                dataset=dataset,
            )
        )

    elif normalized_mode in {
        "rollout",
        "multi_step",
        "multistep",
    }:
        predictions = simulate_rollout(
            parameters=parameters,
            dataset=dataset,
        )

    else:
        raise ValueError(
            "simulation_mode must be either "
            "'one_step' or 'rollout'."
        )

    envelope_residual = (
        predictions["T_env_error"]
        .to_numpy(dtype=np.float64)
        * np.sqrt(envelope_weight)
    )

    zone_residual = (
        predictions["T_zone_error"]
        .to_numpy(dtype=np.float64)
        * np.sqrt(zone_weight)
    )

    residual = np.concatenate(
        [
            envelope_residual,
            zone_residual,
        ]
    )

    if normalize_by_samples:
        residual = residual / np.sqrt(
            max(len(residual), 1)
        )

    if not np.all(np.isfinite(residual)):
        raise FloatingPointError(
            "Residual vector contains non-finite values."
        )

    return residual


def compute_loss(
    parameters: ParameterInput,
    dataset: pd.DataFrame,
    envelope_weight: float = 1.0,
    zone_weight: float = 1.0,
    simulation_mode: str = "one_step",
    return_rmse: bool = False,
    integration_substeps: Optional[int] = None,
    method: Optional[str] = None,
    **kwargs: Any,
) -> float:
    """
    Calculate scalar weighted loss.

    By default, this returns mean squared error over the weighted
    residual vector. Set return_rmse=True to return weighted RMSE.
    """

    residual = compute_residual_vector(
        parameters=parameters,
        dataset=dataset,
        envelope_weight=(
            envelope_weight
        ),
        zone_weight=zone_weight,
        simulation_mode=(
            simulation_mode
        ),
        normalize_by_samples=False,
        integration_substeps=(
            integration_substeps
        ),
        method=method,
        **kwargs,
    )

    mean_squared_error = float(
        np.mean(
            residual ** 2
        )
    )

    if return_rmse:
        return float(
            np.sqrt(
                mean_squared_error
            )
        )

    return mean_squared_error


# ============================================================
# Controller-state consistency diagnostics
# ============================================================

def calculate_controller_state_errors(
    predictions: pd.DataFrame,
    dataset: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compare predicted final PI states with states stored in the dataset.

    This is useful for confirming exact environment reproduction.
    """

    result = predictions.copy()

    optional_comparisons = (
        (
            "integral_error_end",
            "integral_error_predicted_end",
            "integral_error_end_error",
        ),
        (
            "prev_zat_sp_end",
            "prev_zat_sp_predicted_end",
            "prev_zat_sp_end_error",
        ),
        (
            "damper_signal_end",
            "damper_signal_predicted_end",
            "damper_signal_end_error",
        ),
        (
            "m_fan_end",
            "m_fan_predicted_end",
            "m_fan_end_error",
        ),
    )

    for (
        measured_column,
        predicted_column,
        error_column,
    ) in optional_comparisons:
        if (
            measured_column in dataset.columns
            and predicted_column
            in result.columns
        ):
            measured = dataset[
                measured_column
            ].to_numpy(
                dtype=np.float64
            )

            predicted = result[
                predicted_column
            ].to_numpy(
                dtype=np.float64
            )

            result[
                f"{measured_column}_measured"
            ] = measured

            result[error_column] = (
                predicted
                - measured
            )

    return result


# ============================================================
# Simple executable sanity check
# ============================================================

def _print_metrics(
    title: str,
    metrics: SimulationMetrics,
) -> None:
    """Print metrics in a readable format."""

    print(title)
    print("-" * 78)
    print(
        "Envelope RMSE: "
        f"{metrics.envelope_rmse:.12g} °C"
    )
    print(
        "Zone RMSE:     "
        f"{metrics.zone_rmse:.12g} °C"
    )
    print(
        "Combined RMSE: "
        f"{metrics.combined_rmse:.12g} °C"
    )
    print(
        "Envelope bias: "
        f"{metrics.envelope_bias:.12g} °C"
    )
    print(
        "Zone bias:     "
        f"{metrics.zone_bias:.12g} °C"
    )
    print()


def _standalone_sanity_check() -> None:
    """
    Run a true-parameter reproduction check when this file is executed.

    This assumes the updated dataset has already been collected.
    """

    dataset_path = (
        Path("results")
        / "identification"
        / "identification_data.csv"
    )

    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {dataset_path}\n"
            "Run collect_identification_data.py first."
        )

    true_parameters = (
        ThreeRTwoCParameters(
            C_env=3.1996e6,
            C_air=3.5187e5,
            R_rc=0.00706,
            R_oe=0.02707,
            R_er=0.00369,
        )
    )

    dataset = (
        load_identification_dataset(
            dataset_path
        )
    )

    one_step = (
        simulate_one_step_predictions(
            parameters=true_parameters,
            dataset=dataset,
        )
    )

    rollout = simulate_rollout(
        parameters=true_parameters,
        dataset=dataset,
    )

    one_step_metrics = (
        calculate_simulation_metrics(
            one_step
        )
    )

    rollout_metrics = (
        calculate_simulation_metrics(
            rollout
        )
    )

    print("=" * 78)
    print("EXACT TRUE-PARAMETER REPRODUCTION CHECK")
    print("=" * 78)
    print(f"Dataset: {dataset_path}")
    print(f"Rows:    {len(dataset)}")
    print()

    _print_metrics(
        "One-step prediction",
        one_step_metrics,
    )

    _print_metrics(
        "Sequential rollout",
        rollout_metrics,
    )

    diagnostic_directory = (
        Path("results")
        / "identification"
        / "exact_model_diagnostic"
    )

    diagnostic_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    one_step_with_controller = (
        calculate_controller_state_errors(
            predictions=one_step,
            dataset=dataset,
        )
    )

    one_step_with_controller.to_csv(
        diagnostic_directory
        / "true_parameters_one_step.csv",
        index=False,
    )

    rollout.to_csv(
        diagnostic_directory
        / "true_parameters_rollout.csv",
        index=False,
    )

    print(
        "Results saved to: "
        f"{diagnostic_directory}"
    )

    tolerance = 1.0e-4

    if (
        one_step_metrics.combined_rmse
        <= tolerance
    ):
        print()
        print(
            "PASS: The exact model reproduces the "
            "environment transition."
        )
    else:
        print()
        print(
            "FAIL: True parameters still do not reproduce "
            "the collected transitions."
        )
        print(
            "Inspect controller-state errors and verify that "
            "the new dataset was regenerated."
        )


if __name__ == "__main__":
    _standalone_sanity_check()