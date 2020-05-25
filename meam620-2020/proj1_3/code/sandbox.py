import inspect
import json
import matplotlib as mpl
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from scipy.spatial.transform import Rotation
import time
# import rosbag

from flightsim.animate import animate
from flightsim.axes3ds import Axes3Ds
from flightsim.crazyflie_params import quad_params
from flightsim.simulate import Quadrotor, simulate, ExitStatus
from flightsim.world import World

from proj1_3.code.occupancy_map import OccupancyMap
# from proj1_3.code.se3_control import SE3Control
# from proj1_3.code.se3_control_shane import SE3Control
# from proj1_3.code.world_traj import WorldTraj
from proj1_3.code.world_traj_cvx import WorldTraj


t_step = 1/500
# Improve figure display on high DPI screens.
# mpl.rcParams['figure.dpi'] = 200

# Choose a test example file. You should write your own example files too!
# filename = '../util/test_window.json'
# filename = '../util/test_maze_map1.json'
# filename = '../util/test_lab_2.json'
filename = '../util/test_over_under.json'
# filename = '../util/mymap.json'

# Load the test example.
file = Path(inspect.getsourcefile(lambda:0)).parent.resolve() / '..' / 'util' / filename
world = World.from_file(file)          # World boundary and obstacles.
start  = world.world['start']          # Start point, shape=(3,)
goal   = world.world['goal']           # Goal point, shape=(3,)

# This object defines the quadrotor dynamical model and should not be changed.
quadrotor = Quadrotor(quad_params)
robot_radius = 0.27

# Your SE3Control object (from project 1-1).
my_se3_control = SE3Control(quad_params)

# Your MapTraj object. This behaves like the trajectory function you wrote in
# project 1-1, except instead of giving it waypoints you give it the world,
# start, and goal.
planning_start_time = time.time()
my_world_traj = WorldTraj(world, start, goal)
planning_end_time = time.time()

# Help debug issues you may encounter with your choice of resolution and margin
# by plotting the occupancy grid after inflation by margin. THIS IS VERY SLOW!!
# fig = plt.figure('world')
# ax = Axes3Ds(fig)
# world.draw(ax)
# fig = plt.figure('occupancy grid')
# ax = Axes3Ds(fig)
# resolution = SET YOUR RESOLUTION HERE
# margin = SET YOUR MARGIN HERE
# oc = OccupancyMap(world, resolution, margin)
# oc.draw(ax)
# ax.plot([start[0]], [start[1]], [start[2]], 'go', markersize=10, markeredgewidth=3, markerfacecolor='none')
# ax.plot( [goal[0]],  [goal[1]],  [goal[2]], 'ro', markersize=10, markeredgewidth=3, markerfacecolor='none')
# plt.show()

# Set simulation parameters.
t_final = 20
initial_state = {'x': start,
                 'v': (0, 0, 0),
                 'q': (0, 0, 0, 1), # [i,j,k,w]
                 'w': (0, 0, 0)}

# Perform simulation.
#
# This function performs the numerical simulation.  It returns arrays reporting
# the quadrotor state, the control outputs calculated by your controller, and
# the flat outputs calculated by you trajectory.

print()
print('Simulate.')
(sim_time, state, control, flat, exit) = simulate(initial_state,
                                              quadrotor,
                                              my_se3_control,
                                              my_world_traj,
                                              t_final)
print(exit.value)

# Print results.
#
# Only goal reached, collision test, and flight time are used for grading.

collision_pts = world.path_collisions(state['x'], robot_radius)

stopped_at_goal = (exit == ExitStatus.COMPLETE) and np.linalg.norm(state['x'][-1] - goal) <= 0.05
no_collision = collision_pts.size == 0
flight_time = sim_time[-1]
flight_distance = np.sum(np.linalg.norm(np.diff(state['x'], axis=0),axis=1))
planning_time = planning_end_time - planning_start_time

print()
print(f"Results:")
print(f"  No Collision:    {'pass' if no_collision else 'FAIL'}")
print(f"  Stopped at Goal: {'pass' if stopped_at_goal else 'FAIL'}")
print(f"  Flight time:     {flight_time:.1f} seconds")
# print(f"  Flight distance: {flight_distance:.1f} meters")
print(f"  Flight distance: {flight_distance:} meters")
print(f"  Planning time:   {planning_time:.1f} seconds")
if not no_collision:
    print()
    print(f"  The robot collided at location {collision_pts[0]}!")

# Plot Results
#
# You will need to make plots to debug your quadrotor.
# Here are some example of plots that may be useful.

# Visualize the original dense path from A*, your sparse waypoints, and the
# smooth trajectory.
# rosbag:
bagfile = 'Map2.bag'


# bagfile = 'proj1_4_lim_wstep.bag'

# Load the flight data from bag.
ac_state = {}
ac_control = {}
with rosbag.Bag(bagfile) as bag:
# with pyrosbag.Bag(bagfile) as bag:
    odometry = np.array([
        np.array([t.to_sec() - bag.get_start_time(),
        msg.pose.pose.position.x, msg.pose.pose.position.y, msg.pose.pose.position.z,
        msg.twist.twist.linear.x, msg.twist.twist.linear.y, msg.twist.twist.linear.z,
        msg.twist.twist.angular.x, msg.twist.twist.angular.y, msg.twist.twist.angular.z,
        msg.pose.pose.orientation.x, msg.pose.pose.orientation.y, msg.pose.pose.orientation.z, msg.pose.pose.orientation.w])
            for (_, msg, t) in bag.read_messages(topics=['odom'])])
    vicon_time = odometry[:, 0]
    # print(len(vicon_time))
    ac_state['x'] = odometry[:, 1:4]
    ac_state['v'] = odometry[:, 4:7]
    ac_state['w'] = odometry[:, 7:10]
    ac_state['q'] = odometry[:, 10:15]

    commands = np.array([
        np.array([t.to_sec() - bag.get_start_time(),
        msg.linear.z, msg.linear.y, msg.linear.x])
            for (_, msg, t) in bag.read_messages(topics=['so3cmd_to_crazyflie/cmd_vel_fast'])])
    command_time = commands[:, 0]
    # print(len(command_time))
    c1 = -0.6709 # Coefficients to convert thrust PWM to Newtons.
    c2 = 0.1932
    c3 = 13.0652
    ac_control['cmd_thrust'] = (((commands[:, 1]/60000 - c1) / c2)**2 - c3)/1000*9.81
    ac_control['cmd_q'] = Rotation.from_euler('zyx', np.transpose([commands[:, 2], commands[:, 3],
                                                         np.zeros(commands[:, 2].shape)]), degrees=True).as_quat()


ac_flight_distance = np.sum(np.linalg.norm(np.diff(ac_state['x'], axis=0),axis=1))
print("ac_flight_distance: {}".format(ac_flight_distance))

# world = World.empty([-2, 2, -2, 2, 0, 4])
# world = World.empty([-2, 2, -2, 2, 0, 2])


# fig = plt.figure('3D Path (created by rosbag)')
# ax = Axes3Ds(fig)
# ax.set_zlabel("z (m)")
# world.draw(ax)
# world.draw_points(ax, state['x'], color='blue', markersize=4)
# ax.legend(handles=[
#     Line2D([], [], color='blue', linestyle='', marker='.', markersize=4, label='Flight')],
#     loc='upper right')
# ax.set_xlabel("x (m)")
# ax.set_ylabel("y (m)")
# ax.set_zlabel("z (m)")

######################################






fig = plt.figure('A* Path, Waypoints, Trajectory, Actual Flight')
ax = Axes3Ds(fig)
world.draw(ax)
ax.plot([start[0]], [start[1]], [start[2]], 'go', markersize=16, markeredgewidth=3, markerfacecolor='none')
ax.plot( [goal[0]],  [goal[1]],  [goal[2]], 'ro', markersize=16, markeredgewidth=3, markerfacecolor='none')
if hasattr(my_world_traj, 'path'):
    if my_world_traj.path is not None:
        world.draw_line(ax, my_world_traj.path, color='red', linewidth=1)
else:
    print("Have you set \'self.path\' in WorldTraj.__init__?")
if hasattr(my_world_traj, 'points'):
    if my_world_traj.points is not None:
        world.draw_points(ax, my_world_traj.points, color='purple', markersize=8)
else:
    print("Have you set \'self.points\' in WorldTraj.__init__?")
world.draw_line(ax, flat['x'], color='black', linewidth=2)
world.draw_points(ax, ac_state['x'], color='blue', markersize=4)
# ax.legend(handles=[
#     Line2D([], [], color='blue', linestyle='', marker='.', markersize=4, label='Flight')],
#     loc='upper right')
ax.set_xlabel("x (m)")
ax.set_ylabel("y (m)")
ax.set_zlabel("z (m)")
ax.legend(handles=[
    Line2D([], [], color='red', linewidth=1, label='Dense A* Path'),
    Line2D([], [], color='purple', linestyle='', marker='.', markersize=8, label='Sparse Waypoints'),
    Line2D([], [], color='black', linewidth=2, label='Trajectory'),
    Line2D([], [], color='blue', linestyle='', marker='.', markersize=4, label='Actual Flight')],
    loc='upper right')

# Position and Velocity vs. Time
(fig, axes) = plt.subplots(nrows=2, ncols=1, sharex=True, num='Position & Velocity vs Time')
x = state['x']
x_des = flat['x']
ax = axes[0]
ax.plot(sim_time, x_des[:,0], 'r', sim_time, x_des[:,1], 'g', sim_time, x_des[:,2], 'b')
ax.plot(sim_time, x[:,0], 'r.',    sim_time, x[:,1], 'g.',    sim_time, x[:,2], 'b.')
ax.legend(('x', 'y', 'z'), loc='upper right')
ax.set_ylabel('position, m')
ax.grid('major')
ax.set_title('Position')
v = state['v']
v_des = flat['x_dot']
ax = axes[1]
ax.plot(sim_time, v_des[:,0], 'r', sim_time, v_des[:,1], 'g', sim_time, v_des[:,2], 'b')
ax.plot(sim_time, v[:,0], 'r.',    sim_time, v[:,1], 'g.',    sim_time, v[:,2], 'b.')
ax.legend(('x', 'y', 'z'), loc='upper right')
ax.set_ylabel('velocity, m/s')
ax.set_xlabel('time, s')
ax.set_title('Velocity')
ax.grid('major')

# Acceleration and Jerk vs. Time
# (fig, axes) = plt.subplots(nrows=2, ncols=1, sharex=True, num='Acceleration & Jerk vs Time')
# # compute acc and jerk:
# # print(state['v'].shape)
# L_velo = state['v'].shape[0]
# acc_x_lst, acc_y_lst, acc_z_lst = [], [], []
# acc_x_lst.append(0)
# acc_y_lst.append(0)
# acc_z_lst.append(0)
# for i_acc in range(L_velo-1):
#     velo_1, velo_2 = state['v'][i_acc], state['v'][i_acc+1]
#     acc = (velo_2 - velo_1) / t_step
#     acc_x_lst.append(acc[0])
#     acc_y_lst.append(acc[1])
#     acc_z_lst.append(acc[2])
#
# L_acc = len(acc_x_lst)
# jerk_x_lst, jerk_y_lst, jerk_z_lst = [0], [0], [0]
# for i_jerk in range(L_acc-1):
#     acc_x_1, acc_x_2 = acc_x_lst[i_jerk], acc_x_lst[i_jerk+1]
#     acc_y_1, acc_y_2 = acc_y_lst[i_jerk], acc_y_lst[i_jerk+1]
#     acc_z_1, acc_z_2 = acc_z_lst[i_jerk], acc_z_lst[i_jerk+1]
#     jerk_x = (acc_x_2 - acc_x_1) / t_step
#     jerk_y = (acc_y_2 - acc_y_1) / t_step
#     jerk_z = (acc_z_2 - acc_z_1) / t_step
#     jerk_x_lst.append(jerk_x)
#     jerk_y_lst.append(jerk_y)
#     jerk_z_lst.append(jerk_z)

# acc_des = flat['x_ddot']
# ax = axes[0]
# ax.plot(sim_time, acc_des[:,0], 'r', sim_time, acc_des[:,1], 'g', sim_time, acc_des[:,2], 'b')
# ax.plot(sim_time, acc_x_lst, 'r.',    sim_time, acc_y_lst, 'g.',    sim_time, acc_z_lst, 'b.')
# ax.legend(('x', 'y', 'z'), loc='upper right')
# ax.set_ylabel('acceleration, m/(s^2)')
# ax.grid('major')
# ax.set_title('Acceleration')
# jerk_des = flat['x_dddot']
# ax = axes[1]
# ax.plot(sim_time, jerk_des[:,0], 'r', sim_time, jerk_des[:,1], 'g', sim_time, jerk_des[:,2], 'b')
# ax.plot(sim_time, jerk_x_lst, 'r.',    sim_time, jerk_y_lst, 'g.',    sim_time, jerk_z_lst, 'b.')
# ax.legend(('x', 'y', 'z'), loc='upper right')
# ax.set_ylabel('jerk, m/(s^3)')
# ax.set_xlabel('time, s')
# ax.set_title('Jerk')
# ax.grid('major')



# Orientation and Angular Velocity vs. Time
(fig, axes) = plt.subplots(nrows=2, ncols=1, sharex=True, num='Orientation vs Time')
q_des = control['cmd_q']
q = state['q']
ax = axes[0]
ax.plot(sim_time, q_des[:,0], 'r', sim_time, q_des[:,1], 'g', sim_time, q_des[:,2], 'b', sim_time, q_des[:,3], 'k')
ax.plot(sim_time, q[:,0], 'r.',    sim_time, q[:,1], 'g.',    sim_time, q[:,2], 'b.',    sim_time, q[:,3],     'k.')
ax.legend(('i', 'j', 'k', 'w'), loc='upper right')
ax.set_ylabel('quaternion')
ax.set_xlabel('time, s')
ax.grid('major')
w = state['w']
ax = axes[1]
ax.plot(sim_time, w[:,0], 'r.', sim_time, w[:,1], 'g.', sim_time, w[:,2], 'b.')
ax.legend(('x', 'y', 'z'), loc='upper right')
ax.set_ylabel('angular velocity, rad/s')
ax.set_xlabel('time, s')
ax.grid('major')

# Commands vs. Time
(fig, axes) = plt.subplots(nrows=3, ncols=1, sharex=True, num='Commands vs Time')
s = control['cmd_motor_speeds']
ax = axes[0]
ax.plot(sim_time, s[:,0], 'r.', sim_time, s[:,1], 'g.', sim_time, s[:,2], 'b.', sim_time, s[:,3], 'k.')
ax.legend(('1', '2', '3', '4'), loc='upper right')
ax.set_ylabel('motor speeds, rad/s')
ax.grid('major')
ax.set_title('Commands')
M = control['cmd_moment']
ax = axes[1]
ax.plot(sim_time, M[:,0], 'r.', sim_time, M[:,1], 'g.', sim_time, M[:,2], 'b.')
ax.legend(('x', 'y', 'z'), loc='upper right')
ax.set_ylabel('moment, N*m')
ax.grid('major')
T = control['cmd_thrust']
ax = axes[2]
ax.plot(sim_time, T, 'k.')
ax.set_ylabel('thrust, N')
ax.set_xlabel('time, s')
ax.grid('major')

# 3D Paths
fig = plt.figure('3D Path')
ax = Axes3Ds(fig)
world.draw(ax)
ax.plot([start[0]], [start[1]], [start[2]], 'go', markersize=16, markeredgewidth=3, markerfacecolor='none')
ax.plot( [goal[0]],  [goal[1]],  [goal[2]], 'ro', markersize=16, markeredgewidth=3, markerfacecolor='none')
world.draw_line(ax, flat['x'], color='black', linewidth=2)
world.draw_points(ax, state['x'], color='blue', markersize=4)
if collision_pts.size > 0:
    ax.plot(collision_pts[0,[0]], collision_pts[0,[1]], collision_pts[0,[2]], 'rx', markersize=36, markeredgewidth=4)
ax.legend(handles=[
    Line2D([], [], color='black', linewidth=2, label='Trajectory'),
    Line2D([], [], color='blue', linestyle='', marker='.', markersize=4, label='Flight')],
    loc='upper right')


# Animation (Slow)
#
# Instead of viewing the animation live, you may provide a .mp4 filename to save.

R = Rotation.from_quat(state['q']).as_dcm()
animate(sim_time, state['x'], R, world=world, filename=None, show_axes=True)



plt.show()
