import numpy as np
import math
from scipy.spatial.distance import cdist

class WaypointTraj(object):
    """

    """
    def __init__(self, points):
        """
        This is the constructor for the Trajectory object. A fresh trajectory
        object will be constructed before each mission. For a waypoint
        trajectory, the input argument is an array of 3D destination
        coordinates. You are free to choose the times of arrival and the path
        taken between the points in any way you like.

        You should initialize parameters and pre-compute values such as
        polynomial coefficients here.

        Inputs:
            points, (N, 3) array of N waypoint coordinates in 3D
        """

        # STUDENT CODE HERE
        max_accel= 0.5  #m/s^2


        # Make sure there is at least 2 points in points
        if len(points)<2:
            np.append(points,points[0])
        n_seg=len(points)-1

        self._t_knot_list=[0]
        self._seg_list=[]

        # for each segment design trajectory

        # initial and final time for each segment, initialize v0 to 0
        v0 = 0
        vf = 0
        for seg_index in range(n_seg):
            distance = np.linalg.norm(points[seg_index]-points[seg_index+1])
            # use heuristic to calculate final velocity of the segment
            if seg_index is not (n_seg-1):
                next_distance = np.linalg.norm(points[seg_index+1]-points[seg_index+2])
                cos_angle = np.dot(points[seg_index+2]-points[seg_index+1], points[seg_index+1]-points[seg_index])/distance/next_distance
                # if the next segment is in the opposite direction of current section, or orthogonal, no final velocity
                if cos_angle <= 0:
                    vf = 0
                else:
                    # project max accel onto next segment with a safety factor of cos angle to reduce velocity transfer
                    comp_accel = 0# cos_angle**4 * max_accel
                    # calculate final velocity assuming next trajectory has zero final velocity and is constant decelerating
                    vf = math.sqrt(2 * comp_accel * next_distance)
                    # if the final velocity is too high for current trajectory, cap it
                    if vf > math.sqrt(2 * max_accel * distance):
                        vf = math.sqrt(2 * max_accel * distance)
            else:
                vf = 0
                cos_angle = 0
            # minimum time bang bang trajectory calculation with non zero end velocities (solution using mathematica)
            time_traj = - (v0 + vf -math.sqrt(2)*math.sqrt(v0**2+vf**2+2*max_accel*distance))/max_accel
            time_switch = (max_accel*time_traj-v0+vf)/2/max_accel
            t_knot=[0,time_switch, time_traj] # setup knot points
            accel_coeff=[[0,0,max_accel],[0,0,-max_accel]]
            vel_coeff  =[[0,max_accel,v0], [0,-max_accel,max_accel*time_switch+v0]]
            pos_coeff  =[[1/2*max_accel, v0, 0],[-1/2*max_accel, max_accel*time_switch+v0, max_accel*time_switch**2/2+time_switch*v0]]

            # add trajectory to list of trajectories
            self._t_knot_list.append(self._t_knot_list[seg_index]+time_traj)
            self._seg_list.append(TrajSegment(time_traj, t_knot, accel_coeff, vel_coeff, pos_coeff, [points[seg_index], points[seg_index+1]]))

            # initial velocity for next segment is final velocity projected onto next segment
            v0 = vf*cos_angle

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

        # initialize the return value
        x        = np.zeros((3,))
        x_dot    = np.zeros((3,))
        x_ddot   = np.zeros((3,))
        x_dddot  = np.zeros((3,))
        x_ddddot = np.zeros((3,))
        yaw = 0
        yaw_dot = 0
        flat_output = { 'x':x, 'x_dot':x_dot, 'x_ddot':x_ddot, 'x_dddot':x_dddot, 'x_ddddot':x_ddddot,
                        'yaw':yaw, 'yaw_dot':yaw_dot}

        # STUDENT CODE HERE

        # constrain time
        if t < 0:
            t = 0

        # figure out which segment we are in, and adjust time to segment time
        for segment_index in range(len(self._t_knot_list)-1):
            if t>=self._t_knot_list[segment_index] and t<=self._t_knot_list[segment_index+1]:
                tau=t-self._t_knot_list[segment_index]
                flat_output = self._seg_list[segment_index].get_output(tau)
        # if the time is past end of trajectory call the final segment
        if t >= self._t_knot_list[-1]:
            flat_output = self._seg_list[-1].get_output(t)
        return flat_output


class TrajSegment(object):
    """
    Spline object for the straight line, accel and velocity limit trajectory
    """
    def __init__(self, time_traj, t_knot, accel_coeff, vel_coeff, pos_coeff,points):
        """
        Inputs:
            time_traj, length of time of the trajectory
            t_knot, list of the time of the knot points
            accel_coeff, list of list of acceleration coefficients for the polynomials
            points, 2x3, start and end point
        """
        self._time_traj=time_traj
        self._t_knot=t_knot
        self._accel_coeff=accel_coeff
        self._vel_coeff=vel_coeff
        self._pos_coeff=pos_coeff

        self._start=points[0]

        # make sure distance is large enough before normalizing vector of travel to get direction
        if np.linalg.norm(points[1]-points[0])>0.00001:
            self._dir=(points[1]-points[0])/np.linalg.norm(points[1]-points[0])
        else:
            self._dir=np.zeros(3)


    def _eval_poly(self,t, coefficient):
        """
        Evaluates a polynomial based on coefficient

        Inputs:
            coefficient, 3x1 list, [x^2, x, 1]
        Ouptuts:
            value of polynomial at time t
        """
        return coefficient[0]*pow(t, 2) + coefficient[1]*t+coefficient[2]

    def get_output(self, t):
        """
        Calculates flat output of trajectory based on trajectory time t

        Inputs:
            t, time from 0 to time_traj that spline is evaluated at

        Outputs:
            flat_output
        """
        x        = np.zeros(3)
        x_dot    = np.zeros((3,))
        x_ddot   = np.zeros((3,))
        x_dddot  = np.zeros((3,))
        x_ddddot = np.zeros((3,))
        yaw = 0
        yaw_dot = 0

        # save initial time passed in before constraining it
        t_int = t
        # constrain time
        if t > self._time_traj:
            t = self._time_traj
        else:
            if t < 0:
                t = 0

        # find the spline segment and evaluate the polynomial using the segment time
        for segment_index in range(len(self._t_knot)-1):
            # evaluate spline based on constrained time
            if t>=self._t_knot[segment_index] and t<=self._t_knot[segment_index+1]:
                tau=t-self._t_knot[segment_index]
                r        = self._eval_poly(tau,self._pos_coeff[segment_index])
                r_dot    = self._eval_poly(tau,self._vel_coeff[segment_index])
                r_dotdot = self._eval_poly(tau,self._accel_coeff[segment_index])
                x        = r*self._dir+self._start
                x_dot    = r_dot*self._dir
                x_ddot = r_dotdot*self._dir
                # if the trajectory has ended, (based on unconstrain time), acceleration is 0
                if t_int >= self._time_traj:
                    x_ddot = 0
                break
        flat_output = { 'x':x, 'x_dot':x_dot, 'x_ddot':x_ddot, 'x_dddot':x_dddot, 'x_ddddot':x_ddddot,
                        'yaw':yaw, 'yaw_dot':yaw_dot}
        return flat_output


