"""Physical and controller parameters of the drones, shared by the controllers, the motor model and the
asset builder (``scripts/assets/build_flycrane_asset.py``).

Taken from the hook-grasp project (same lab, same OmniNxt drone), adapted for the flycrane (3 drones carrying
a payload): only OMNINXT.omega_max differs, see the comment there.

Pure Python (no torch / Isaac import) so the asset builder can load this file standalone.

Frames: positions are given in the drone's design frame (x forward, y left, z up). For the OmniNxt this is
the origin and axes of the user's Fusion assembly. ``*_rel_com`` properties express them relative to the
centre of mass, which is where the controllers and the URDF base link live.

Rotor order and spin (both drones): 0 front-left (+x,+y) CW, 1 front-right (+x,-y) CCW,
2 rear-right (-x,-y) CW, 3 rear-left (-x,+y) CCW (seen from above). ``rotor_dir`` is the sign of the yaw
reaction torque each rotor puts on the body (+1 for CW props).
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class DroneParams:
    name: str
    # --- rigid body ---
    mass: float                                   # kg, total drone mass (used by the controllers)
    com: tuple[float, float, float]               # m, centre of mass in the design frame
    inertia_diag: tuple[float, float, float]      # kg m^2, Ixx Iyy Izz at the CoM (physics + INDI)
    inertia_offdiag: tuple[float, float, float]   # kg m^2, Ixy Ixz Iyz at the CoM (physics only; INDI is diagonal)
    # --- rotors ---
    rotor_pos: tuple[tuple[float, float, float], ...]  # m, 4 propeller hubs in the design frame
    rotor_dir: tuple[float, float, float, float]       # yaw reaction-torque sign per rotor
    c_t: float                                    # N/(rad/s)^2, thrust = c_t * omega^2
    kappa: float                                  # m, c_q / c_t
    omega_min: float                              # rad/s (declared only, not enforced)
    omega_max: float                              # rad/s, sets thrust_max = c_t * omega_max^2
    tau_up: float                                 # s, motor first-order time constant (spin-up)
    tau_down: float                               # s, motor first-order time constant (spin-down)
    motor_inertia: float                          # kg m^2, rotor inertia (I_p, declared only, not modelled)
    # --- wire ---
    wire_attach: tuple[float, float, float]       # m, wire anchor in the design frame
    # --- geometric controller gains (ACCBR uses kp_att_* and kp_rate) ---
    kp_att_xy: float
    kp_att_z: float
    kp_rate: tuple[float, float, float]
    # thrust_max override (N per rotor); None -> c_t * omega_max^2
    thrust_max_override: float | None = None

    @property
    def c_q(self) -> float:
        """N m/(rad/s)^2, rotor drag torque = c_q * omega^2."""
        return self.kappa * self.c_t

    @property
    def thrust_max(self) -> float:
        """N, maximum thrust per rotor."""
        if self.thrust_max_override is not None:
            return self.thrust_max_override
        return self.c_t * self.omega_max**2

    @property
    def rotor_pos_rel_com(self) -> tuple[tuple[float, float, float], ...]:
        return tuple(tuple(p[k] - self.com[k] for k in range(3)) for p in self.rotor_pos)

    @property
    def wire_attach_rel_com(self) -> tuple[float, float, float]:
        return tuple(self.wire_attach[k] - self.com[k] for k in range(3))

    def hover_omega(self, gravity: float = 9.8066) -> float:
        """rad/s, rotor speed at which the 4 rotors carry the drone's weight."""
        return math.sqrt(self.mass * gravity / (4.0 * self.c_t))


# Falcon: the values that were hard-coded in indi.py / geometric.py / motor_model.py (unchanged behaviour for
# the hover / flycrane tasks). Its physics lives in falcon_data/falcon.urdf, not generated from this.
_FALCON_ARM = 0.10606601717798213 * math.sin(math.radians(45.0))  # 0.075 m, l * sin(beta)
FALCON = DroneParams(
    name="falcon",
    mass=0.6017,
    com=(0.0, 0.0, 0.03),
    inertia_diag=(0.00164, 0.00184, 0.0030),
    inertia_offdiag=(0.0, 0.0, 0.0),
    rotor_pos=(
        (_FALCON_ARM, _FALCON_ARM, 0.05),
        (_FALCON_ARM, -_FALCON_ARM, 0.05),
        (-_FALCON_ARM, -_FALCON_ARM, 0.05),
        (-_FALCON_ARM, _FALCON_ARM, 0.05),
    ),
    rotor_dir=(1.0, -1.0, 1.0, -1.0),
    c_t=1.562522e-06,
    kappa=0.022,
    omega_min=150.0,
    omega_max=2800.0,
    tau_up=0.033,
    tau_down=0.033,
    motor_inertia=9.3575e-6,
    wire_attach=(0.0, 0.0, 0.0),
    kp_att_xy=150.0,
    kp_att_z=5.0,
    kp_rate=(25.0, 25.0, 8.0),
    thrust_max_override=6.25,
)

# OmniNxt (the lab's real drone). Sources, 2026-11-09:
#   mass, CoM, inertia off-diagonal, rotor hubs, wire anchor: Fusion assembly (inertia "at centre of mass",
#     off-diagonals taken as tensor elements -integral(xy dm)); wire anchored under the CoM on the frame's
#     bottom face (z = 0);
#   inertia diagonal, kp_att_xy, kp_att_z, kp_rate: supervisor's onboard INDI tuning;
#   c_t, kappa: (= onboard INDI_CT, INDI_KM); tau_up/down.
#   motor_inertia: whole rotor from the CAD (1.549e-6, Jack's INDI table); not used by the simulation
#     (the motor model is a first-order lag, tau_up / tau_down).
#   omega_max: set so that c_t * omega_max^2 = 6 N per rotor, the limit Jack asked for (2026-11-13).
#     Only the resulting thrust limit is enforced (INDI clamps each rotor to thrust_max). For reference:
#     the hook-grasp version uses 45,800 rpm (11.3 N); our thrust stand reached 30,700 rpm at 90 % throttle.
#   wire_attach: assumed under the CoM on the frame's bottom face (as in the hook-grasp task): TO CONFIRM.
OMNINXT = DroneParams(
    name="omninxt",
    mass=0.723,
    com=(0.004456, 0.00011, 0.01524),
    inertia_diag=(0.0009, 0.00135, 0.001861),
    inertia_offdiag=(3.537953e-6, -2.4987806e-5, 2.34938e-7),
    rotor_pos=(
        (0.05350, 0.05350, -0.018412),
        (0.05350, -0.05350, -0.018412),
        (-0.05350, -0.05350, -0.018412),
        (-0.05350, 0.05350, -0.018412),
    ),
    rotor_dir=(1.0, -1.0, 1.0, -1.0),
    c_t=4.9e-7,
    kappa=0.014,
    omega_min=150.0,
    omega_max=math.sqrt(6.0 / 4.9e-7),   # 3,499 rad/s (33,400 rpm): the speed that gives Jack's 6 N per rotor
    tau_up=0.025,
    tau_down=0.037,
    motor_inertia=1.549e-6,   # whole rotor, CAD (Jack's INDI table); not used by the simulation
    wire_attach=(0.004456, 0.00011, 0.0),
    kp_att_xy=60.0,
    kp_att_z=16.0,
    kp_rate=(15.5, 15.5, 8.0),
)

DRONES = {p.name: p for p in (FALCON, OMNINXT)}


def get_drone_params(name: str) -> DroneParams:
    try:
        return DRONES[name]
    except KeyError as e:
        raise ValueError(f"unknown drone '{name}', expected one of {sorted(DRONES)}") from e