import math
import heapq
import numpy as np

class AStarPlanner:
    def __init__(self, map_size):
        self.map_size = map_size
        self.last_plan = None

    def is_in_bounds(self, pos):
        x, y = pos
        return 0 <= x < self.map_size[0] and 0 <= y < self.map_size[1]

    def heuristic(self, a, b):
        return math.hypot(b[0] - a[0], b[1] - a[1])

    def a_star(self, start, goal, cost_map):
        open_set = [(0, start)]
        came_from = {}
        g_score = {start: 0}

        while open_set:
            _, current = heapq.heappop(open_set)
            if self.heuristic(current, goal) < 1.5:
                goal = current
                break

            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    if dx == 0 and dy == 0:
                        continue
                    neighbor = (current[0] + dx, current[1] + dy)
                    ix, iy = int(neighbor[0]), int(neighbor[1])

                    if not self.is_in_bounds((ix, iy)) or cost_map[ix, iy] >= 253:
                        continue

                    move_cost = self.heuristic(current, neighbor)
                    tentative_g = g_score[current] + move_cost + cost_map[ix, iy]

                    if neighbor not in g_score or tentative_g < g_score[neighbor]:
                        g_score[neighbor] = tentative_g
                        f_score = tentative_g + self.heuristic(neighbor, goal)
                        heapq.heappush(open_set, (f_score, neighbor))
                        came_from[neighbor] = current

        if goal not in came_from:
            return []
        path = [goal]
        while path[-1] != start:
            path.append(came_from[path[-1]])
        path.reverse()
        return path

    def plan(self, agent, problem_env):
        start = agent['pos']
        goal = problem_env.goal
        cost_map = problem_env.map.get_cost_map()
        path = self.a_star(start, goal, cost_map)
        return path