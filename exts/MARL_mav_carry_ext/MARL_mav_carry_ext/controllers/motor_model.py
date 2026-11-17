import torch

from MARL_mav_carry_ext.controllers.drone_params import FALCON, DroneParams


class RotorMotor:
    def __init__(self, num_envs: int, init_omega: torch.Tensor, params: DroneParams = FALCON):
        # motor parameters
        self.init_omega = init_omega
        self.num_envs = num_envs
        self.motor_omega_min = params.omega_min
        self.motor_omega_max = params.omega_max
        self.tau_up = params.tau_up
        self.tau_down = params.tau_down
        self.motor_inertia = params.motor_inertia

        # rotor parameters
        self.thrust_map = torch.tensor([params.c_t, 0.0, 0.0])
        self.torque_map = torch.tensor([params.c_q, 0.0, 0.0])  # c_q = kappa * c_t (Falcon: 3.4375484e-08)

        self.current_omega = init_omega
        self.direction = torch.tensor(params.rotor_dir, device="cuda")

    def get_motor_thrusts_moments(self, target_rates: torch.Tensor, sampling_time: float):
        """
        Compute filtered thrusts and angular velocities for the motor.

        Args:
            target_rates: Desired rotational rate (omega) from the controller.
            sampling_time: Time interval between two consecutive calls.

        Returns:
            filtered_thrusts: Filtered thrust values.
        """
        # Check if body rate is going up or down
        tau = torch.where(target_rates > self.current_omega, self.tau_up, self.tau_down)
        alpha = torch.exp(-sampling_time / tau)

        # Update current omega
        self.current_omega = alpha * self.current_omega + (1 - alpha) * target_rates

        # Compute thrusts based on current omega
        thrusts = self.thrust_map[0] * self.current_omega**2

        # Compute moments based on current omega
        moments = self.torque_map[0] * self.current_omega**2 * self.direction

        return thrusts, moments

    def reset(self, env_ids):
        self.current_omega[env_ids] = self.init_omega[env_ids]