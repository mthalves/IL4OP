import math
import matplotlib.pyplot as plt
import matplotlib.patches as ptc
import numpy as np
import numpy.ma as ma
from scipy.ndimage import distance_transform_edt

def compute_dist(a,b):
    return math.hypot(a[0] - b[0], a[1] - b[1])

def bresenham(x0, y0, x1, y1):
    points = []
    dx, dy = abs(x1 - x0), abs(y1 - y0)
    x, y = x0, y0
    sx, sy = 1 if x0 < x1 else -1, 1 if y0 < y1 else -1
    if dx > dy:
        err = dx / 2.0
        while x != x1:
            points.append((x, y))
            err -= dy
            if err < 0:
                y += sy
                err += dx
            x += sx
    else:
        err = dy / 2.0
        while y != y1:
            points.append((x, y))
            err -= dx
            if err < 0:
                x += sx
                err += dy
            y += sy
    points.append((x1, y1))
    return points


class InflationMap:

    def __init__(
        self,
        map_size_w          :tuple[float,float],
        resolution          :float=0.25,
        z_min               :float=0.1,
        z_max               :float=1.0,
        confirm_threshold   :int  =2,
        inscribed_radius    :float=0.5,
        inflation_radius    :float=0.1,
        cost_scaling_factor :float=0.5,
    ):
        self.resolution = resolution
        self.map_size_w = map_size_w
        self.map_size = (int(map_size_w[0] / resolution), int(map_size_w[1] / resolution))

        self.z_min = z_min
        self.z_max = z_max
        self.confirm_threshold = confirm_threshold

        self.inscribed_radius = inscribed_radius
        self.inflation_radius = inflation_radius
        self.cost_scaling_factor = cost_scaling_factor
        self.max_cost = 254

        self.hit_count_map = np.zeros(self.map_size, dtype=np.int16)
        self.obstacle_map = np.zeros_like(self.hit_count_map, dtype=np.uint8)
        self.cost_map = np.zeros_like(self.obstacle_map, dtype=np.float32)

        self._fig = None

    def reset(self):
        self.hit_count_map = np.zeros(self.map_size, dtype=np.int16)
        self.obstacle_map = np.zeros_like(self.hit_count_map, dtype=np.uint8)
        self.cost_map = np.zeros_like(self.obstacle_map, dtype=np.float32)

        plt.close('all')
        plt.ion()
        self._fig, self._ax = None, None
        self._im = None

    ###
    ### SUPPORTIVE METHODS
    ###
    def map_to_world(self, x, y):
        """Converts map coordinates to world coordinates"""
        return x * self.resolution, y * self.resolution
    
    def world_to_map(self, x, y):
        """Converts world coordinates to map coordinates"""
        return int(x / self.resolution), int(y / self.resolution)

    def is_in_bounds(self, pos):
        """Check if position is within the map boundaries map."""
        x, y = pos
        return 0 <= x < self.map_size[0] and 0 <= y < self.map_size[1]
    
    def is_visible(self, eye, obj, visibility_radius=np.inf, threshold=0.99):
        """Check if the object is visible based on the line between two world
         points. If the line passes through an inflated cost map cell (considering
         a threshold) or it is out the visibility radius, it is not visible."""
        x0, y0 = eye
        x1, y1 = obj

        # checking if the object is in the visibility radius
        if compute_dist(eye,obj) > visibility_radius:
            return False

        # checking the inflated cells
        line_cells = bresenham(x0, y0, x1, y1)
        for x, y in line_cells:
            if self.cost_map[x, y] > threshold*self.max_cost:
                return False
        return True
    
    def is_free_space(self, pos):
        return self.cost_map[pos[0], pos[1]] < 0.95*self.max_cost

    def compute_orientation(self, robot_pos, target_point):
        """Computes robot orientation given the map robot and target position"""
        x0, y0 = int(robot_pos[0]), int(robot_pos[1])
        x1, y1 = int(target_point[0]), int(target_point[1])
        angle = math.atan2(y1 - y0, x1 - x0)
        return angle

    ###
    ### MAP GENERATION METHODS
    ###
    def update_with_lidar(self, robot_pos_w, lidar_hits_w, max_dist=None):
        """Updates cost map based on the world lidar hits and robot position"""
        rx, ry = self.world_to_map(*robot_pos_w)
        max_dist_sq = (max_dist)**2 if max_dist is not None else np.inf

        for hit in lidar_hits_w:
            hx, hy, hz = hit
            if not (self.z_min <= hz <= self.z_max):
                continue

            mx, my = self.world_to_map(hx, hy)
            points = bresenham(rx, ry, mx, my)
            if not points:
                continue

            for x, y in points[:-1]:
                # Only check bounds once
                if not self.is_in_bounds((x, y)):
                    continue

                # Distance check (squared)
                dx, dy = x - rx, y - ry
                if dx**2 + dy**2 > max_dist_sq:
                    continue

                # Clip to avoid underflow
                self.hit_count_map[x, y] = max(self.hit_count_map[x, y] - 1, 0)

            # Final hit point
            if self.is_in_bounds((mx, my)):
                self.hit_count_map[mx, my] += 1

        # Update obstacle map only once
        self.obstacle_map = (self.hit_count_map >= self.confirm_threshold).astype(np.uint8)

    def compute_cost_map(self):
        dist = distance_transform_edt(1 - self.obstacle_map) * self.resolution
        cost = np.zeros_like(dist, dtype=np.float32)
        cost = self.max_cost * np.exp(-self.cost_scaling_factor * (dist - self.inscribed_radius))
        cost[dist <= self.inscribed_radius] = self.max_cost
        cost[dist > self.inflation_radius] = 0
        self.cost_map = cost

    def get_obstacle_map(self):
        return np.copy(self.obstacle_map)
    
    def get_cost_map(self):
        return np.copy(self.cost_map)

    ###
    ### VISUALIZATION METHODS
    ###    
    def visualize(self, robot=None, path=None, tasks=None, memory_map=None, step=None):
        # Throttle
        if step is not None and step % 10 != 0:
            return

        # Inverting cost map for visualization
        viz_cost_map = np.zeros(self.map_size[::-1])
        viz_memory_map = np.zeros(self.map_size[::-1])
        for x in range(self.map_size[0]):
            for y in range(self.map_size[1]):
                viz_cost_map[y,x] = self.cost_map[x,y]
                if memory_map is not None:
                    viz_memory_map[y,x] = memory_map[x,y]
        
        # Init figure
        if self._fig is None:
            self._fig, self._ax = plt.subplots(figsize=(6, 6))
            self._ax.set_xlim(0, viz_cost_map.shape[0]-1)
            self._ax.set_ylim(0, viz_cost_map.shape[1]-1)

            # Base cost map (obstacles + inflation)
            self._im = self._ax.imshow(
                viz_cost_map,
                cmap='inferno', origin='lower',
                vmin=0, vmax=255, alpha=1., zorder=1
            )

            # Memory overlay (visited cells faint white)
            self._memory_overlay = self._ax.imshow(
                np.zeros_like(viz_cost_map),
                cmap='gray', origin='lower', 
                vmin=0, vmax=1, alpha=0.4, zorder=3
            )

            # Planning elements
            self._path_line, = self._ax.plot([], [], 'r.-', label="Planned Path", zorder=6)
            self._robot_patch = None
            self._visibility_circle = None  
            self._task_markers, = self._ax.plot([], [], 'bs', label="Tasks", zorder=7)

            # Title, grid, and colorbar
            self._ax.set_title("Planning Map")
            self._ax.set_xticks(range(0, viz_cost_map.shape[0]), minor=True)
            self._ax.set_yticks(range(0, viz_cost_map.shape[1]), minor=True)
            self._ax.grid(which='minor', alpha=0.8)
            self._ax.grid(which='major', alpha=0.9)
            self._ax.legend()
            self._colorbar = self._fig.colorbar(self._im, ax=self._ax, fraction=0.04, pad=0.04)

        # === Update overlays ===
        self._im.set_data(viz_cost_map)
        if viz_memory_map is not None:
            visited_mask = (viz_memory_map > 0)
            self._memory_overlay.set_data(visited_mask.astype(float))

        # === Robot marker ===
        if robot:
            mx, my = robot['pos']
            radius_px = robot['radius']
            visibility_radius_px = robot['visibility_radius']

            # Visibility circle
            if self._visibility_circle is None:
                self._visibility_circle = ptc.Circle(
                    (mx, my), visibility_radius_px,
                    color='cyan', fill=False, linestyle='--', linewidth=1.5, alpha=0.7, zorder=5
                )
                self._ax.add_patch(self._visibility_circle)
            else:
                self._visibility_circle.center = (mx, my)

            # Robot patch
            if self._robot_patch is None:
                self._robot_patch = ptc.Circle((mx, my), radius_px, color='g', zorder=6)
                self._ax.add_patch(self._robot_patch)
            else:
                self._robot_patch.center = (mx, my)

        # === Task markers ===
        if tasks:
            tasks_pos = [tpos for tpos in tasks.values()]
            tx, ty = zip(*tasks_pos)
            self._task_markers.set_data(tx, ty)
        else:
            self._task_markers.set_data([], [])

        # === Path line ===
        if path:
            px, py = zip(*path)
            self._path_line.set_data(px, py)
        else:
            self._path_line.set_data([], [])

        # Redraw
        self._fig.canvas.draw()
        self._fig.canvas.flush_events()