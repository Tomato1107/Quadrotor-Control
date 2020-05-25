import numpy as np
from scipy.spatial.transform import Rotation

# Get Rotation Matrix from quaternions:
def quat2Rot(q):
    q0, q1, q2, q3 = q[3], q[0], q[1], q[2]
    R = np.array([
        [1-2*q2**2-2*q3**2, 2*q1*q2-2*q0*q3, 2*q1*q3+2*q0*q2],
        [2*q1*q2+2*q0*q3, 1-2*q1**2-2*q3**2, 2*q2*q3-2*q0*q1],
        [2*q1*q3-2*q0*q2, 2*q2*q3+2*q0*q1, 1-2*q1**2-2*q2**2]])
    return R




class SE3Control(object):
    """

    """
    def __init__(self, quad_params):
        """
        This is the constructor for the SE3Control object. You may instead
        initialize any parameters, control gain values, or private state here.

        For grading purposes the controller is always initialized with one input
        argument: the quadrotor's physical parameters. If you add any additional
        input arguments for testing purposes, you must provide good default
        values!

        Parameters:
            quad_params, dict with keys specified by crazyflie_params.py

        """

        # Quadrotor physical parameters.
        self.mass            = quad_params['mass'] # kg
        self.Ixx             = quad_params['Ixx']  # kg*m^2
        self.Iyy             = quad_params['Iyy']  # kg*m^2
        self.Izz             = quad_params['Izz']  # kg*m^2
        self.arm_length      = quad_params['arm_length'] # meters
        self.rotor_speed_min = quad_params['rotor_speed_min'] # rad/s
        self.rotor_speed_max = quad_params['rotor_speed_max'] # rad/s
        self.k_thrust        = quad_params['k_thrust'] # N/(rad/s)**2
        self.k_drag          = quad_params['k_drag']   # Nm/(rad/s)**2

        # You may define any additional constants you like including control gains.
        self.inertia = np.diag(np.array([self.Ixx, self.Iyy, self.Izz])) # kg*m^2
        self.g = 9.81 # m/s^2

        # STUDENT CODE HERE
        self.gamma = self.k_drag / self.k_thrust
        # Kp & Kd gains:
        # self.K_p = np.diag(np.array([15, 8, 15]))
        # self.K_d = np.diag(np.array([10.7, 4, 10]))

        # new params:
        # self.K_p = np.diag(np.array([8.5, 8.5, 10])) # origin
        self.K_p = np.diag(np.array([8.5, 8.5, 10]))
        # self.K_d = np.diag(np.array([5, 5, 5])) # origin
        self.K_d = np.diag(np.array([3, 3, 5]))
        # Kr & Kw gains:
        # self.K_r = np.diag(np.array([2500, 2500, 20]))
        # self.K_w = np.diag(np.array([300, 300, 7.55]))

        # new params:
        self.K_r = np.diag(np.array([2500, 2500, 400]))
        self.K_w = np.diag(np.array([60, 60, 50]))
 
    def update(self, t, state, flat_output):
        """
        This function receives the current time, true state, and desired flat
        outputs. It returns the command inputs.

        Inputs:
            t, present time in seconds
            state, a dict describing the present state with keys
                x, position, m
                v, linear velocity, m/s
                q, quaternion [i,j,k,w]
                w, angular velocity, rad/s
            flat_output, a dict describing the present desired flat outputs with keys
                x,        position, m
                x_dot,    velocity, m/s
                x_ddot,   acceleration, m/s**2
                x_dddot,  jerk, m/s**3
                x_ddddot, snap, m/s**4
                yaw,      yaw angle, rad
                yaw_dot,  yaw rate, rad/s

        Outputs:
            control_input, a dict describing the present computed control inputs with keys
                cmd_motor_speeds, rad/s
                cmd_thrust, N (for debugging and laboratory; not used by simulator)
                cmd_moment, N*m (for debugging; not used by simulator)
                cmd_q, quaternion [i,j,k,w] (for laboratory; not used by simulator)
        """
        cmd_motor_speeds = np.zeros((4,))
        cmd_thrust = 0
        cmd_moment = np.zeros((3,))
        cmd_q = np.zeros((4,))

        # STUDENT CODE HERE
        Error_velo = state['v'] - flat_output['x_dot']
        Error_pos = state['x'] - flat_output['x']

        # equation (26):
        r_acc_des = -np.dot(self.K_d, Error_velo) - np.dot(self.K_p, Error_pos)

        # equation (27) compute Force desired:
        F_des = self.mass * r_acc_des + np.array([0, 0, self.mass * self.g])

        # Rotation Matrix & b3 transpose:
        Rot_mat = quat2Rot(state['q'])
        b3_T = np.dot(np.array([[0,0,1]]), np.transpose(Rot_mat))

        # equation (29) get u1:
        u1 = np.dot(b3_T, F_des)

        # equation (30) get b3_des:
        b3_des = F_des / (np.sqrt(sum(np.square(F_des))))

        # equation (31) get yaw direction:
        psi_T = flat_output['yaw']
        a_psi = np.array([np.cos(psi_T), np.sin(psi_T), 0])

        # equation (32) get b2_des:
        b2_des = np.cross(b3_des, a_psi) / np.sqrt(sum(np.square(np.cross(b3_des, a_psi))))

        # equation (33) get R_des:
        R_des_23 = np.concatenate((b2_des.reshape(3,1),b3_des.reshape(3,1)),axis=1)
        R_des_1 = np.cross(b2_des, b3_des)
        R_des = np.concatenate((R_des_1.reshape(3,1),R_des_23),axis=1)

        # equation (34) get error_R:
        error_R_mat = 1/2 * np.dot(np.transpose(R_des),Rot_mat) - 1/2 * np.dot(np.transpose(Rot_mat),R_des)
        # print(error_R_mat)
        error_R = np.array([error_R_mat[2][1], error_R_mat[0][2], error_R_mat[1][0]])
        # print(error_R)
        # print(error_R_mat)

        # equation (35) get u2:
        error_w = state['w']
        u2 = np.dot(self.inertia, np.dot(-self.K_r,error_R)) - np.dot(self.inertia, np.dot(self.K_w,error_w))

        # output:
        cmd_thrust = u1
        cmd_moment = u2
        # u_mat = np.concatenate((u1,u2.reshape(3,1)),axis=0)
        u_mat = np.concatenate((u1,u2),axis=0)
        # print(u_mat.shape)
        L = self.arm_length
        gamma = self.gamma
        Coef_mat = np.array([
            [1,1,1,1],
            [0,L,0,-L],
            [-L,0,L,0],
            [gamma,-gamma,gamma,-gamma]])

        F_mat = np.dot(np.linalg.inv(Coef_mat),u_mat)
        F_vec = F_mat.flatten()
        for f in range(len(F_vec)):
            if F_vec[f] <= 0.001:
                F_vec[f] = 0.0

        # F = k_f * w^2:
        w_vec = np.sqrt(F_vec / self.k_thrust)
        max_speed, min_speed = self.rotor_speed_max, self.rotor_speed_min
        # constrain output motor speed to (min, max):
        for s in range(len(w_vec)):
            if w_vec[s] >= max_speed:
                w_vec[s] = max_speed
            elif w_vec[s] <= min_speed:
                w_vec[s] = min_speed

        cmd_motor_speeds = w_vec


        control_input = {'cmd_motor_speeds':cmd_motor_speeds,
                         'cmd_thrust':cmd_thrust,
                         'cmd_moment':cmd_moment,
                         'cmd_q':cmd_q}
        return control_input
