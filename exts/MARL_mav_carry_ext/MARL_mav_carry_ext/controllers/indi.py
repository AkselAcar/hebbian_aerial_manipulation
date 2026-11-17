import torch

from MARL_mav_carry_ext.controllers.drone_params import FALCON, DroneParams
from MARL_mav_carry_ext.controllers.utils import LowPassFilter

from isaaclab.utils.math import quat_inv, quat_apply


class IndiController:
    def __init__(self, num_envs: int, params: DroneParams = FALCON):
        self.num_envs = num_envs
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.params = params

        # allocation matrix: [collective thrust; roll; pitch; yaw] = G_1 @ rotor thrusts, with each rotor's arm
        # (x_i, y_i) measured from the CoM (roll = sum y_i T_i, pitch = -sum x_i T_i, yaw = kappa sum dir_i T_i).
        # For the Falcon this is exactly the former l*sin(beta) / l*cos(beta) layout (l = 0.10607 m, beta = 45 deg).
        self.kappa = params.kappa
        arms = torch.tensor(params.rotor_pos_rel_com, device=self.device)[:, :2]  # (4, 2) x_i, y_i
        rotor_dir = torch.tensor(params.rotor_dir, device=self.device)
        self.G_1 = torch.stack(
            [torch.ones(4, device=self.device), arms[:, 1], -arms[:, 0], self.kappa * rotor_dir], dim=0
        )
        self.G_1_inv = torch.linalg.inv(self.G_1)

        self.motor_inertia_z = params.motor_inertia  # [kgm^2] (I_p, G_2 is not used)
        self.G_2 = torch.zeros((4, 4), device=self.device)
        self.G_2[3, :] = self.motor_inertia_z * rotor_dir
        self.motor_omega_min = params.omega_min  # [rad/s]
        self.motor_omega_max = params.omega_max  # [rad/s]
        self.thrust_min = 0.0
        self.thrust_max = params.thrust_max  # [N]
        self.thrust_min_collective = 0.0
        self.thrust_max_collective = self.thrust_max * 4  # [N]
        self.thrust_map = torch.tensor([params.c_t, 0.0, 0.0], device=self.device)

        self.falcon_mass = params.mass  # kg (name kept from the Falcon code)
        self.inertia_mat = torch.diag(torch.tensor(params.inertia_diag, device=self.device))
        self.rope_offset = -0.03
        self.p_offset = torch.tensor([[0.0, 0.0, self.rope_offset]] * self.num_envs, device=self.device)

        # TODO REMOVE
        # self.kp = torch.tensor([100.0, 100.0, 10.0], device=self.device)

        # low pass filters
        self.filter_sampling_frequency = torch.full(
            (self.num_envs, 1), 300.0, device=self.device
        )  # filter frequency, same as control frequency (Hz)
        self.filter_cutoff_frequency = torch.full(
            (self.num_envs, 1), 12.0, device=self.device
        )  # accelerometer filter cut-off frequency (Hz)
        self.filter_init_value_mot = torch.full((self.num_envs, 4), 0.0, device=self.device)
        self.filter_init_value_rate = torch.full((self.num_envs, 3), 0.0, device=self.device)

        self.filterMot_ = LowPassFilter(
            self.filter_cutoff_frequency, self.filter_sampling_frequency, self.filter_init_value_mot
        )
        self.filterRate_ = LowPassFilter(
            self.filter_cutoff_frequency, self.filter_sampling_frequency, self.filter_init_value_rate
        )

        self.debug = True
        if self.debug:
            self.filtered_ang_acc = torch.zeros((self.num_envs, 3), device=self.device)
            self.unfiltered_mot = torch.zeros((self.num_envs, 4), device=self.device)
            self.filtered_mot = torch.zeros((self.num_envs, 4), device=self.device)

        # INDI disturbance residual (body frame, 6-dim): the external force/moment the drone MEASURES
        # beyond its commanded control -- i.e. what the hook+wire imparts, sensed onboard (IMU) WITHOUT any
        # hook/wall position. [ang(3): alpha_cmd - omega_dot (the INDI angular increment, line ~117);
        # lin(3): R^T(acc_cmd - a_meas) (commanded vs measured specific force, rotated to body)]. Exposed
        # here so the hook-grasp env can feed it to its hook-blind actor (onboard-measurable hook signal).
        self.disturbance_residual = torch.zeros((self.num_envs, 6), device=self.device)

    def getCommand(
        self,
        state: dict,
        actions: torch.tensor,
        alpha_cmd: torch.tensor,
        acc_cmd: torch.tensor,
        acc_load: torch.tensor,
    ) -> torch.tensor:
        forces = actions.sum(-1)
        filtered_forces = self.filterMot_.add(forces)
        self.filterRate_.add(state["ang_vel"])
        ang_acc_filtered = self.filterRate_.derivative()

        if self.debug:
            self.filtered_ang_acc = ang_acc_filtered
            self.unfiltered_mot = forces
            self.filtered_mot = filtered_forces

        omega = quat_apply(quat_inv(state["quat"]), state["ang_vel"])  # body rates # normally from IMU
        omega_dot = quat_apply(
            quat_inv(state["quat"]), state["ang_acc"]
        )  # body accelerations # normally from derivative filtered body rate
        # INDI disturbance residual (body frame): angular = the INDI increment (alpha_cmd - omega_dot);
        # linear = commanded minus measured specific force, rotated into the body frame. Carries the hook/wire
        # external wrench (onboard-measurable, position-free) for the optional actor observation.
        self.disturbance_residual = torch.cat(
            (alpha_cmd - omega_dot, quat_apply(quat_inv(state["quat"]), acc_cmd - state["lin_acc"])),
            dim=-1,
        )
        tau = torch.matmul(self.G_1, forces.transpose(0, 1)).transpose(0, 1)[:, 1:]  # torque commands
        mu = torch.zeros((self.num_envs, 4), device=self.device)
        collective_thrust_des_magntiude = torch.norm(acc_cmd, dim=1) * self.falcon_mass
        mu[:, 0] = torch.clamp(collective_thrust_des_magntiude, self.thrust_min_collective, self.thrust_max_collective)
        mu_ndi = mu

        moments = self.inertia_mat.matmul(alpha_cmd.transpose(0, 1)).transpose(0, 1) + torch.linalg.cross(
            omega, self.inertia_mat.matmul(omega.transpose(0, 1)).transpose(0, 1)
        )  # - torch.linalg.cross(
        # self.p_offset, quat_apply(quat_inv(state["quat"]), acc_load * self.falcon_mass)) # M_load in body frame

        mu_ndi[:, 1:] = moments
        mu[:, 1:] = tau + self.inertia_mat.matmul((alpha_cmd - omega_dot).transpose(0, 1)).transpose(0, 1)

        # without heading control
        mu[:, 3] = mu_ndi[:, 3]
        thrusts = self.G_1_inv.matmul(mu.transpose(0, 1))
        thrusts = torch.clamp(thrusts, self.thrust_min, self.thrust_max)

        rotor_speeds = torch.sqrt(thrusts / self.thrust_map[0]).transpose(0, 1)

        return rotor_speeds

    # def getCommand(
    #     self,
    #     state: dict,
    #     actions: torch.tensor,
    #     setpoint: dict,
    # ) -> torch.tensor:
    #     forces = actions.sum(-1)
    #     # filtered_forces = self.filterMot_.add(forces)
    #     # self.filterRate_.add(state["ang_vel"])
    #     # ang_acc_filtered = self.filterRate_.derivative()

    #     # if self.debug:
    #     #     self.filtered_ang_acc = ang_acc_filtered
    #     #     self.unfiltered_mot = forces
    #     #     self.filtered_mot = filtered_forces

    #     omega = quat_apply(quat_inv(state["quat"]), state["ang_vel"])  # body rates # normally from IMU
    #     alpha_cmd = self.kp * (setpoint["body_rates"] - omega)
    #     acc_cmd = setpoint["cthrust"]
    #     omega_dot = quat_apply(
    #         quat_inv(state["quat"]), state["ang_acc"]
    #     )  # body accelerations # normally from derivative filtered body rate
    #     tau = torch.matmul(self.G_1, forces.transpose(0, 1)).transpose(0, 1)[:, 1:]  # torque commands
    #     mu = torch.zeros((self.num_envs, 4), device=self.device)
    #     # collective_thrust_des_magntiude = torch.norm(acc_cmd, dim=1) * self.falcon_mass
    #     collective_thrust_des_magntiude = acc_cmd * self.falcon_mass
    #     mu[:, 0] = torch.clamp(collective_thrust_des_magntiude, self.thrust_min_collective, self.thrust_max_collective)
    #     mu_ndi = mu

    #     moments = self.inertia_mat.matmul(alpha_cmd.transpose(0, 1)).transpose(0, 1) + torch.linalg.cross(
    #         omega, self.inertia_mat.matmul(omega.transpose(0, 1)).transpose(0, 1)
    #     )  # - torch.linalg.cross(
    #     # self.p_offset, quat_apply(quat_inv(state["quat"]), acc_load * self.falcon_mass)) # M_load in body frame

    #     mu_ndi[:, 1:] = moments
    #     mu[:, 1:] = tau + self.inertia_mat.matmul((alpha_cmd - omega_dot).transpose(0, 1)).transpose(0, 1)

    #     # without heading control
    #     mu[:, 3] = mu_ndi[:, 3]
    #     thrusts = self.G_1_inv.matmul(mu.transpose(0, 1))
    #     thrusts = torch.clamp(thrusts, self.thrust_min, self.thrust_max)

    #     rotor_speeds = torch.sqrt(thrusts / self.thrust_map[0]).transpose(0, 1)

    #     return rotor_speeds

    def reset(self, env_ids):
        self.filterMot_.reset(env_ids)
        self.filterRate_.reset(env_ids)
        self.disturbance_residual[env_ids] = 0.0
        # pass