import numpy as np

from proj1_3.code.graph_search import graph_search
from cvxopt import solvers, matrix
import cvxopt
cvxopt.solvers.options['show_progress'] = False


# compute angle between vectors: (points are arrays)
def cmp_ang(point1, point2):
    x_dir = np.array([1, 0, 0])
    y_dir = np.array([0, 1, 0])
    z_dir = np.array([0, 0, 1])
    pt2_pt1_abs = np.sqrt(sum(np.square(point2 - point1)))
    x_ang = np.arccos(np.dot(point2 - point1, x_dir) / pt2_pt1_abs)
    y_ang = np.arccos(np.dot(point2 - point1, y_dir) / pt2_pt1_abs)
    z_ang = np.arccos(np.dot(point2 - point1, z_dir) / pt2_pt1_abs)

    return x_ang, y_ang, z_ang



# remove unnecessary points:
def rm_inter_pt(path): # path is array
    ori_path = path
    cp_path = np.copy(path)
    del_idx_lst = []
    for i in range(cp_path.shape[0]-2):
        pt_1, pt_2, pt_3 = cp_path[i], cp_path[i+1], cp_path[i+2]
        p1p2 = pt_2 - pt_1
        p2p3 = pt_3 - pt_2
        p1p2_norm = np.sqrt(sum(np.square(p1p2)))
        p2p3_norm = np.sqrt(sum(np.square(p2p3)))
        dot_val = np.dot(p1p2,p2p3)
        cos_theta = dot_val / (p1p2_norm * p2p3_norm)
        if cos_theta >= 0.9: # remove unnecessary point:
            del_idx_lst.append(i+1)
            # new_path = np.delete(ori_path,i+1,axis=0)
    new_path = np.delete(ori_path,del_idx_lst,axis=0) # remove points
    # print(del_idx_lst)
    return new_path


# compute optimization:
def solve_opt(point_0, point_1, t):
    c_list = []
    scale_mat = np.array([[t**5], [t**4], [t**3], [t**2], [t], [1.0]])
    # scale_mat = np.array([[1 / (t ** 5), 1 / (t ** 4), 1 / (t ** 3), 1 / (t ** 2), 1 / t, 1.0]])
    Q = 2*matrix([[3600.0, 1440.0, 360.0, 0.0, 0.0, 0.0],
                  [1440.0, 576.0, 144.0, 0.0, 0.0, 0.0],
                  [360.0, 144.0, 36.0, 0.0, 0.0, 0.0],
                  [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                  [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                  [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])

    p = matrix([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    G = matrix([[5.0,-5.0,20.0,-20.0],
                [4.0,-4.0,12.0,-12.0],
                [3.0,-3.0,6.0,-6.0],
                [2.0,-2.0,2.0,-2.0],
                [1.0,-1.0,0.0,0.0],
                [0.0,0.0,0.0,0.0]])
    # G = matrix([[5.0, 4.0, 3.0, 2.0, 1.0, 0.0],
    #             [-5.0, -4.0, -3.0, -2.0, -1.0, -0.0]])
    # print(G.size)
    h = matrix([5.0, 0.0, 5.0, -0.5])

    A = matrix([[0.0,0.0,0.0,1.0,5.0,20.0],
                [0.0,0.0,0.0,1.0,4.0,12.0],
                [0.0,0.0,0.0,1.0,3.0,6.0],
                [0.0,0.0,2.0,1.0,2.0,2.0],
                [0.0,1.0,0.0,1.0,1.0,0.0],
                [1.0,0.0,0.0,1.0,0.0,0.0]])
    # A = matrix([[0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
    #            [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
    #            [0.0, 0.0, 0.0, 2.0, 0.0, 0.0],
    #            [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
    #            [5.0, 4.0, 3.0, 2.0, 1.0, 0.0],
    #            [20.0, 12.0, 6.0, 2.0, 0.0, 0.0]])
    # for i in range(3): # for loop for x,y,z
    b_x = matrix([point_0[0], 0.0, 0.0, point_1[0], 0.0, 0.0])
    b_y = matrix([point_0[1], 0.0, 0.0, point_1[1], 0.0, 0.0])
    b_z = matrix([point_0[2], 0.0, 0.0, point_1[2], 0.0, 0.0])
    # b_x = matrix([[point_0[0]], [0.0], [0.0], [point_1[0]], [0.0], [0.0]])
    # b_y = matrix([[point_0[1]], [0.0], [0.0], [point_1[1]], [0.0], [0.0]])
    # b_z = matrix([[point_0[2]], [0.0], [0.0], [point_1[2]], [0.0], [0.0]])
    # solve for solutions:
    c_x = np.array(solvers.qp(Q,p,G,h,A,b_x)["x"])
    c_y = np.array(solvers.qp(Q,p,G,h,A,b_y)["x"])
    c_z = np.array(solvers.qp(Q,p,G,h,A,b_z)["x"])

    # rescaling
    new_c_x = scale_mat * c_x
    new_c_y = scale_mat * c_y
    new_c_z = scale_mat * c_z

    return new_c_x, new_c_y, new_c_z

# def revise_T_list(T_list):
#     cp_T_list = T_list[:]
#     for i in range(len(T_list)-1):
#         T0, T1 = T_list[i], T_list[i+1]
#         cp_T_list[i+1] = T1 - T0
#
#     return cp_T_list[1:]




class WorldTraj(object):
    """

    """
    def __init__(self, world, start, goal):
        """
        This is the constructor for the trajectory object. A fresh trajectory
        object will be constructed before each mission. For a world trajectory,
        the input arguments are start and end positions and a world object. You
        are free to choose the path taken in any way you like.

        You should initialize parameters and pre-compute values such as
        polynomial coefficients here.

        Parameters:
            world, World object representing the environment obstacles
            start, xyz position in meters, shape=(3,)
            goal,  xyz position in meters, shape=(3,)

        """

        # You must choose resolution and margin parameters to use for path
        # planning. In the previous project these were provided to you; now you
        # must chose them for yourself. Your may try these default values, but
        # you should experiment with them!
        # self.resolution = np.array([0.25, 0.25, 0.25])
        self.resolution = np.array([0.2, 0.2, 0.2])
        self.margin = 0.3

        # You must store the dense path returned from your Dijkstra or AStar
        # graph search algorithm as an object member. You will need it for
        # debugging, it will be used when plotting results.
        self.path = graph_search(world, self.resolution, self.margin, start, goal, astar=True)
        # print("origin length: {}".format(self.path.shape[0]))
        # remove unnecessary points:
        self.path = np.delete(self.path,1,axis=0)
        # self.path = np.delete(self.path, 2, axis=0)
        self.path = np.delete(self.path,-2,axis=0)
        self.path = rm_inter_pt(self.path)
        # print("new length: {}".format(self.path.shape[0]))

        # You must generate a sparse set of waypoints to fly between. Your
        # original Dijkstra or AStar path probably has too many points that are
        # too close together. Store these waypoints as a class member; you will
        # need it for debugging and it will be used when plotting results.
        # self.points = np.zeros((1,3)) # shape=(n_pts,3)
        self.points = np.delete(self.path, 1, axis=0)
        # self.path = np.delete(self.path, 2, axis=0)
        self.points = np.delete(self.path, -2, axis=0)
        self.points = rm_inter_pt(self.path)
        # Finally, you must compute a trajectory through the waypoints similar
        # to your task in the first project. One possibility is to use the
        # WaypointTraj object you already wrote in the first project. However,
        # you probably need to improve it using techniques we have learned this
        # semester.
        self.start = start
        self.goal = goal
        # STUDENT CODE HERE
        self.velo = 2.5
        self.T_finish = 0.0
        self.Length = 0.0
        self.new_range = 0
        self.traj_x, self.traj_y, self.traj_z, \
        self.velo_x, self.velo_y, self.velo_z, \
        self.acc_x, self.acc_y, self.acc_z, \
        self.jerk_x, self.jerk_y, self.jerk_z = self.get_traj()
        # def get_traj(self):

    def get_traj(self):
        pt_array = self.points
        # print(len(self.points))
        mean_velo = self.velo
        N = pt_array.shape[0]

        self.T_list = [] # store T interval between each segment
        total_L = 0
        for i in range(N):
            if i < N - 1:
                pt_a = pt_array[i]
                pt_b = pt_array[i + 1]
                L = np.sqrt(sum(np.square(pt_a - pt_b)))
                self.T_list.append(L / mean_velo)
                total_L += L
        self.Length = total_L # sum length of all waypoints
        self.T_finish = total_L / mean_velo
        # self.revised_T_list = revise_T_list(self.T_list)
        t_inter = 1000
        # output: traj_list
        traj_list_x, traj_list_y, traj_list_z = [], [], []
        velo_list_x, velo_list_y, velo_list_z = [], [], []
        acc_list_x, acc_list_y, acc_list_z = [], [], []
        jerk_list_x, jerk_list_y, jerk_list_z = [], [], []

        # print(len(self.T_list))
        # print(self.revised_T_list)
        for p_idx in range(N-1):
            p0, p1 = pt_array[p_idx], pt_array[p_idx+1]
            # for delta_t in self.revised_T_list: # time interval list
            delta_t = self.T_list[p_idx]
            c_x, c_y, c_z = solve_opt(p0, p1, delta_t) # get coefficients of polynomial
            c_x, c_y, c_z = c_x.reshape(1,6), c_y.reshape(1,6), c_z.reshape(1,6)
            tt = np.linspace(0.0,1.0,int(t_inter * delta_t)).reshape(1,int(t_inter * delta_t))
            tt_2 = np.power(tt,2)
            tt_3 = np.power(tt,3)
            tt_4 = np.power(tt,4)
            tt_5 = np.power(tt,5)
            tt_0 = np.ones(int(t_inter * delta_t)).reshape(1,int(t_inter * delta_t))
            zero_vec = np.zeros(int(t_inter * delta_t)).reshape(1,int(t_inter * delta_t))
            tt_mat = np.concatenate((tt_5,tt_4,tt_3,tt_2,tt,tt_0),axis=0)
            # print(tt_mat.shape)
            # x_list, y_list, z_list are arrays
            # get position list
            x_list = np.dot(c_x,tt_mat).flatten().tolist()
            y_list = np.dot(c_y,tt_mat).flatten().tolist()
            z_list = np.dot(c_z,tt_mat).flatten().tolist()
            traj_list_x += x_list
            traj_list_y += y_list
            traj_list_z += z_list

            # get velocity list
            tt_mat_v = np.concatenate((5*tt_4, 4*tt_3, 3*tt_2, 2*tt, tt_0, zero_vec),axis=0)
            # tt_mat_v = np.concatenate((zero_vec, 5 * tt_4, 4 * tt_3, 3 * tt_2, 2 * tt, tt_0), axis=0)
            temp_v_x = np.dot(c_x,tt_mat_v).flatten().tolist()
            temp_v_y = np.dot(c_y,tt_mat_v).flatten().tolist()
            temp_v_z = np.dot(c_z,tt_mat_v).flatten().tolist()
            velo_list_x += temp_v_x
            velo_list_y += temp_v_y
            velo_list_z += temp_v_z

            # get acceleration list
            tt_mat_acc = np.concatenate((20*tt_3, 12*tt_2, 6*tt, 2*tt_0, zero_vec, zero_vec),axis=0)
            # tt_mat_acc = np.concatenate((zero_vec, zero_vec, 20*tt_3, 12*tt_2, 6*tt, 2*tt_0),axis=0)

            temp_acc_x = np.dot(c_x,tt_mat_acc).flatten().tolist()
            temp_acc_y = np.dot(c_y, tt_mat_acc).flatten().tolist()
            temp_acc_z = np.dot(c_z, tt_mat_acc).flatten().tolist()
            acc_list_x += temp_acc_x
            acc_list_y += temp_acc_y
            acc_list_z += temp_acc_z

            # get jerk list
            # tt_mat_acc = np.concatenate((zero_vec, zero_vec, zero_vec, 60*tt_2, 24*tt, 6*tt_0),axis=0)
            tt_mat_acc = np.concatenate((60*tt_2, 24*tt, 6*tt_0, zero_vec, zero_vec, zero_vec),axis=0)

            temp_jerk_x = np.dot(c_x, tt_mat_acc).flatten().tolist()
            temp_jerk_y = np.dot(c_y, tt_mat_acc).flatten().tolist()
            temp_jerk_z = np.dot(c_z, tt_mat_acc).flatten().tolist()
            jerk_list_x += temp_jerk_x
            jerk_list_y += temp_jerk_y
            jerk_list_z += temp_jerk_z

        # append final point's state
        traj_list_x.append(self.goal[0])
        traj_list_y.append(self.goal[1])
        traj_list_z.append(self.goal[2])

        velo_list_x.append(0)
        velo_list_y.append(0)
        velo_list_z.append(0)

        acc_list_x.append(0)
        acc_list_y.append(0)
        acc_list_z.append(0)

        jerk_list_x.append(0)
        jerk_list_y.append(0)
        jerk_list_z.append(0)

        # print(traj_list_x[-1], traj_list_y[-1],traj_list_z[-1])
        # print(self.goal)
        # print(traj_list_x[-100:], traj_list_y[-100:], traj_list_z[-100:])
        # print(traj_list_z)
        # print(self.goal)
        return traj_list_x, traj_list_y, traj_list_z,\
               velo_list_x, velo_list_y, velo_list_z,\
               acc_list_x, acc_list_y, acc_list_z,\
               jerk_list_x, jerk_list_y, jerk_list_z



    def update(self, t):
        """
        Given the present time, return the desired flat output and derivatives.

        Inputs
            t, time, s
        Outputs
            flat_output, a dict describing the present desired flat outputs with keys
                x,        position, m
                x_dot,    velocity, m/s
                x_ddot,   acceleration, m/s**2
                x_dddot,  jerk, m/s**3
                x_ddddot, snap, m/s**4
                yaw,      yaw angle, rad
                yaw_dot,  yaw rate, rad/s
        """
        x        = np.zeros((3,))
        x_dot    = np.zeros((3,))
        x_ddot   = np.zeros((3,))
        x_dddot  = np.zeros((3,))
        x_ddddot = np.zeros((3,))
        yaw = 0
        yaw_dot = 0

        # STUDENT CODE HERE
        # print(t)
        traj_x, traj_y, traj_z = self.traj_x, self.traj_y, self.traj_z
        velo_x, velo_y, velo_z = self.velo_x, self.velo_y, self.velo_z
        acc_x, acc_y, acc_z = self.acc_x, self.acc_y, self.acc_z
        jerk_x, jerk_y, jerk_z = self.jerk_x, self.jerk_y, self.jerk_z
        traj_L = len(traj_x)

        if t == float("inf"):
            t = 9999999

        # Test if a single point:
        if self.T_finish == 0:
            get_num = 0
        else:
            get_num = int(t / self.T_finish * traj_L)

        if get_num > traj_L - 1:  # judge overflow
            get_num = traj_L - 1

        x = np.array([traj_x[get_num], traj_y[get_num], traj_z[get_num]])
        x_dot = np.array([velo_x[get_num], velo_y[get_num], velo_z[get_num]])
        x_ddot = np.array([acc_x[get_num], acc_y[get_num], acc_z[get_num]])
        x_dddot = np.array([jerk_x[get_num], jerk_y[get_num], jerk_z[get_num]])

        # print(x_list)
        # zero_r = np.array([[0, 0, 0]])
        # pt_array = self.path
        # if self.pt[0][0] != 0 or self.pt[0][1] != 0 or self.pt[0][2] != 0:
        #     pt_array = np.r_(zero_r, self.pt)

        # velo_list = []
        # for i in range(self.new_range):
        #     if i < self.new_range - 1:
        #         pt1, pt2 = pt_array[i], pt_array[i + 1]
        #         x_ang, y_ang, z_ang = cmp_ang(pt1, pt2)
        #         x_velo = self.velo * np.cos(x_ang)
        #         y_velo = self.velo * np.cos(y_ang)
        #         z_velo = self.velo * np.cos(z_ang)
        #         velo_list.append(np.array([x_velo, y_velo, z_velo]))
        #
        # velo_list_all = gen_velo_all(velo_list, self.new_range)
        # print(get_num)
        # print("velo_list_length: {}".format(len(velo_list_all)))
        # x_dot = velo_list_all[get_num]


        flat_output = { 'x':x, 'x_dot':x_dot, 'x_ddot':x_ddot, 'x_dddot':x_dddot, 'x_ddddot':x_ddddot,
                        'yaw':yaw, 'yaw_dot':yaw_dot}
        return flat_output
