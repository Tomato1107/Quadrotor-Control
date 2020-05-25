from heapq import heappush, heappop  # Recommended.
import numpy as np
# import heapq
from flightsim.world import World
from proj1_3.code.occupancy_map import OccupancyMap # Recommended.


# create a simple PriorityQueue: always pop min
class PriorityQueue:
    def __init__(self):
        self._queue = []
        self._index = 0

    def push(self, item, priority):
        heappush(self._queue, (priority, self._index, item))
        self._index += 1

    def pop(self):
        return heappop(self._queue)[-1]

    def empty(self):
        if len(self._queue) == 0:
            return True
        else:
            return False


# heuristics:
def Heuristics(point1, point2):
    # point1_pos = map.index_to_metric_center(point1_idx)
    # point2_pos = map.index_to_metric_center(point2_idx)

    distance_2 = sum(np.power((point1 - point2), 2))
    distance = np.power(distance_2, 0.5)
    return distance

# 26-connected: p: point(index)
def connect_26(p, map, X_L, Y_L, Z_L):
    # x_L, y_L, z_L = map.shape[0], map.shape[1], map.shape[2]
    # 6 connected:
    L1 = [(p[0]+1,p[1],p[2]), (p[0]-1,p[1],p[2]),
          (p[0],p[1]+1,p[2]), (p[0],p[1]-1,p[2]),
          (p[0],p[1],p[2]+1), (p[0],p[1],p[2]-1)]

    # 18 connected:
    L2 = [(p[0]+1,p[1]+1,p[2]), (p[0]-1,p[1]-1,p[2]),
          (p[0]+1,p[1]-1,p[2]), (p[0]-1,p[1]+1,p[2]),
          (p[0]+1,p[1],p[2]+1), (p[0]-1,p[1],p[2]-1),
          (p[0]+1,p[1],p[2]-1), (p[0]-1,p[1],p[2]+1),
          (p[0],p[1]+1,p[2]+1), (p[0],p[1]-1,p[2]-1),
          (p[0],p[1]+1,p[2]-1), (p[0],p[1]-1,p[2]+1)]

    # 26 connected:
    L3 = [(p[0]+1,p[1]+1,p[2]+1), (p[0]-1,p[1]-1,p[2]-1),
          (p[0]+1,p[1]+1,p[2]-1), (p[0]-1,p[1]-1,p[2]+1),
          (p[0]+1,p[1]-1,p[2]+1), (p[0]-1,p[1]+1,p[2]-1),
          (p[0]-1,p[1]+1,p[2]+1), (p[0]+1,p[1]-1,p[2]-1)]

    L26 = L1 + L2 + L3
    # L26 = L1 + L2
    # print("list all: {}".format(L26))
    # print(len(L26))
    # set26 = set(L26)

    # check if satisfy: x, y, z in constraint
    L_move = L26[:]
    for tup in L_move:
        if map.is_valid_index(tup) and (not map.is_occupied_index(tup)):
            continue
        else:
            L26.remove(tup)
    #
    #
    # for tup in L26:
    #     if tup[0] < 0 or tup[0] >= X_L or tup[1] < 0 or tup[1] >= Y_L or tup[2] < 0 or tup[2] >= Z_L:
    #         L26.remove(tup)
    #
    neighbors = L26

    return neighbors

# create graph:
def create_graph(map, x_l, y_l, z_l):
    # x_L, y_L, z_L = map.shape[0], map.shape[1], map.shape[2]
    graph = {}
    for i in range(x_l):
        for j in range(y_l):
            for k in range(z_l):
                index = (i, j, k)
                neighbors = connect_26(index, map, x_l, y_l, z_l)
                graph[index] = neighbors

    return graph


def graph_search(world, resolution, margin, start, goal, astar):
    """
    Parameters:
        world,      World object representing the environment obstacles
        resolution, xyz resolution in meters for an occupancy map, shape=(3,)
        margin,     minimum allowed distance in meters from path to obstacles.
        start,      xyz position in meters, shape=(3,)
        goal,       xyz position in meters, shape=(3,)
        astar,      if True use A*, else use Dijkstra
    Output:
        path,       xyz position coordinates along the path in meters with
                    shape=(N,3). These are typically the centers of visited
                    voxels of an occupancy map. The first point must be the
                    start and the last point must be the goal. If no path
                    exists, return None.
    """

    # While not required, we have provided an occupancy map you may use or modify.
    occ_map = OccupancyMap(world, resolution, margin)
    # Retrieve the index in the occupancy grid matrix corresponding to a position in space.
    start_index = tuple(occ_map.metric_to_index(start))
    # print("start index: {}".format(start_index))
    goal_index = tuple(occ_map.metric_to_index(goal))
    # print(occ_map.map)
    x_L, y_L, z_L = occ_map.map.shape[0], occ_map.map.shape[1], occ_map.map.shape[2]
    # print(type(start_index))
    # print(goal_index)

    if astar == True:
        flg = 1
    else:
        flg = 0

    frontier = PriorityQueue()
    frontier.push(start_index, 0)

    came_from = {}
    cost_so_far = {}
    came_from[start_index] = start_index
    cost_so_far[start_index] = 0
    graph = create_graph(occ_map, x_L, y_L, z_L)
    # keep track of path using parent dictionary:
    parent_ND = {}
    parent_ND[start_index] = start_index
    # closed = set()
    ALL_INDEX = []
    for i in range(x_L):
        for j in range(y_L):
            for k in range(z_L):
                ALL_INDEX.append((i,j,k))
    closed = set(ALL_INDEX)

    # test graph:
    # print("test neighbors: {}".format(graph[(1,0,0)]))


    # path_idx
    path_idx = []

    while not frontier.empty():
        current_index  = frontier.pop()
        # print("fron: {}".format(frontier._queue))
        if current_index == goal_index:
            temp = current_index
            # print("temp: {}".format(temp))
            # print(came_from)
            while temp != came_from[temp]:
                path_idx.append(temp)
                temp = came_from[temp]
            # print("path: {}".format(path_idx[::-1]))
            # print("path_length: {}".format(len(path_idx[::-1])))
            path_idx.append(start_index)
            out_path_idx = path_idx[::-1]
            # print("out path index: {}".format(out_path_idx))

            out_path = []
            for idx in out_path_idx:
                temp_pos = occ_map.index_to_metric_center(idx)
                temp_pos = temp_pos.tolist()
                out_path.append(temp_pos)
            final_path_temp = np.array(out_path)
            # print(start)
            path_a = np.concatenate((start.reshape(1,3),final_path_temp),axis=0)
            path_b = np.concatenate((path_a,goal.reshape(1,3)),axis=0)
            # final_path_ = path_b
            # print("out path: {}".format(final_path))
            return path_b
        # print(current_index)
        for next_index in graph[tuple(current_index)]:
            # Use position to compute cost & use index to store values:
            # Heuristics represents eu distance:
            current_pos = occ_map.index_to_metric_center(current_index)
            next_pos = occ_map.index_to_metric_center(next_index)
            new_cost = cost_so_far[current_index] + Heuristics(current_pos, next_pos)
            if next_index not in cost_so_far or new_cost < cost_so_far[next_index]:
            # if next_index not in cost_so_far:
                # save path and judge if reach the goal:
                # parent_ND[next_index] = current_index
                # if next_index == goal_index:
                #     temp = next_index
                #     while (temp != parent_ND[temp]):
                #         path_idx.append(temp)
                #         temp = parent_ND[temp]
                #     return list(path_idx[::-1])
                cost_so_far[next_index] = new_cost
                next_pos = occ_map.index_to_metric_center(next_index)
                priority = new_cost + flg * Heuristics(goal, next_pos) # goal is already the position
                frontier.push(next_index, priority)
                came_from[next_index] = current_index

    return None


