import math
import matplotlib.pyplot as plt
import matplotlib.patches as ptc
import numpy as np
import torch

from isaaclab_experiments.src.mapping.utils import bresenham, compute_dist, distance_transform_edt

class ContinuousInflationMap:

    def __init__(
        self,
        map_size_w          : tuple[float, float],
        resolution          : float = 0.1,
        z_min               : float=0.1,
        z_max               : float=1.0,
        confirm_threshold   : int  =2,
        robot_radius        : float = 0.5,
        inflation_radius    : float = 1.0,
        cost_scaling_factor : float = 0.5,
    ):
        # -------------------------------------------------
        # REAL WORLD
        # -------------------------------------------------
        # Map info
        self.map_size_w = map_size_w # real map size
        self.inflation_radius_w = inflation_radius
        self.resolution = resolution # resolution to translate map to grid

        # sensors info
        self.z_min, self.z_max      = z_min, z_max
        self.confirm_threshold      = confirm_threshold

        # Robot info
        self.robot_radius_w         = robot_radius

        # -------------------------------------------------
        # ABSTRACTION
        # -------------------------------------------------
        self.map_size               = (int(map_size_w[0] / resolution),int(map_size_w[1] / resolution)) # grid map size
        self.robot_radius           = int(robot_radius / self.resolution)

        # Discretized Maps
        self.hit_count_map = np.zeros(self.map_size, dtype=np.int16)
        self.obstacle_map           = np.zeros(self.map_size, dtype=np.uint8)
        self.sdf_map                = np.zeros(self.map_size, dtype=np.float32)

        # Inflation map info
        self.inflation_radius       = int(inflation_radius / self.resolution)
        self.cost_scaling_factor    = cost_scaling_factor
        self.max_cost               = 255
        self.eta                    = 0.95 # collision cost threshold (as a fraction of max cost)   

        # Navigation gradients (spatial gradient)
        self.grad_x = None
        self.grad_y = None

        # -------------------------------------------------
        # UTILS
        # -------------------------------------------------
        self.objects_identified = []
        self._fig = None

    def reset(self):
        self.hit_count_map[:] = 0
        self.obstacle_map[:] = 0
        self.sdf_map[:] = 0
        self.grad_x = None
        self.grad_y = None

        plt.close('all')
        plt.ion()
        self._fig, self._ax = None, None
        self._im = None

    # -------------------------------------------------
    # TRANSFORM/DISCRETIZATION
    # -------------------------------------------------

    def map_to_world(self, x, y):
        """Converts map coordinates to world coordinates"""
        return x * self.resolution, y * self.resolution
    
    def world_to_map(self, x, y):
        """Converts world coordinates to map coordinates"""
        return int(math.floor(x / self.resolution)), int(math.floor(y / self.resolution))

    def is_in_bounds(self, pos):
        """Check if position is within the map boundaries map."""
        x, y = pos
        return 0 <= x < self.map_size[0] and 0 <= y < self.map_size[1]

    
    # -------------------------------------------------
    # MAPPING
    # -------------------------------------------------

    def update_with_lidar(self, robot_pos_w, lidar_readings, lidar_meshes=None, max_dist=None):
        rx, ry = robot_pos_w
        max_dist_sq = max_dist**2 if max_dist is not None \
         else 0.5 * (self.map_size_w[0]**2 + self.map_size_w[1]**2)
        

        # --- separating hits per axis ---
        hits = lidar_readings  # (N, 3)
        hx = hits[:, 0]
        hy = hits[:, 1]
        hz = hits[:, 2]


        # --- vectorized filtering ---
        dx = hx - rx
        dy = hy - ry
        dist_sq = dx**2 + dy**2

        valid_mask = (
            torch.isfinite(hx) & torch.isfinite(hy) &                   # finite check
            (hz >= self.z_min) & (hz <= self.z_max) &                   # valid height
            (dist_sq > (self.robot_radius_w + self.resolution) ** 2)    # in a valid sight
        )

        hx = hx[valid_mask]
        hy = hy[valid_mask]
        dist_sq = dist_sq[valid_mask]

        if hx.numel() == 0:
            return

        # --- convert robot position ---
        cx, cy = self.world_to_map(rx, ry)

        # --- process only valid rays ---
        for i in range(hx.shape[0]):
            
            mx, my = self.world_to_map(float(hx[i]), float(hy[i]))

            points = bresenham(cx, cy, mx, my)
            if not points:
                continue

            # --- free space update ---
            for x, y in points[:-1]:

                if not self.is_in_bounds((x, y)):
                    continue

                dx_map = (x - cx)
                dy_map = (y - cy)

                if (dx_map * dx_map + dy_map * dy_map) > (max_dist_sq / (self.resolution ** 2)):
                    continue

                if self.hit_count_map[x, y] > 0:
                    self.hit_count_map[x, y] -= 1

            # --- occupied cell ---
            if self.is_in_bounds((mx, my)):
                self.hit_count_map[mx, my] += 1

        # Update obstacle map only once
        self.obstacle_map = (self.hit_count_map >= self.confirm_threshold).astype(np.uint8)

        if lidar_meshes is not None:
            self.objects_identified = set(lidar_meshes)
            print("Objects identified in the scene:", self.objects_identified)
        
        # updating sdf map
        self.compute_sdf()
            
    # -------------------------------------------------
    # CONTINUOUS SDF
    # -------------------------------------------------

    def compute_sdf(self):

        outside = np.array(distance_transform_edt(1 - self.obstacle_map))
        inside = np.array(distance_transform_edt(self.obstacle_map))

        self.sdf_map = (outside - inside) * self.resolution

        # Compute gradients (optional but very powerful)
        self.grad_x, self.grad_y = np.gradient(self.sdf_map)

    def sdf(self, pos_w):
        # --- 1. convert to continuous map coordinates ---
        mx = pos_w[0] / self.resolution
        my = pos_w[1] / self.resolution

        # --- 2. integer cell ---
        x0 = int(np.floor(mx))
        y0 = int(np.floor(my))

        # --- 3. clamp neighbors ---
        x1 = min(x0 + 1, self.sdf_map.shape[0] - 1)
        y1 = min(y0 + 1, self.sdf_map.shape[1] - 1)

        # --- 4. bounds check ---
        if not self.is_in_bounds((x0, y0)):
            return -self.robot_radius

        # --- 5. fractional offset ---
        dx = mx - x0
        dy = my - y0

        # --- 6. fetch values ---
        d00 = self.sdf_map[x0, y0]
        d10 = self.sdf_map[x1, y0]
        d01 = self.sdf_map[x0, y1]
        d11 = self.sdf_map[x1, y1]

        # --- 7. bilinear interpolation ---
        d0 = d00 * (1 - dx) + d10 * dx
        d1 = d01 * (1 - dx) + d11 * dx

        return (d0 * (1 - dy) + d1 * dy)

    def cost(self, pos_w):
        # sdf cost function
        d = self.sdf(pos_w)

        if d <= self.robot_radius_w:
            return self.max_cost

        if d > self.inflation_radius_w:
            return 0

        return self.max_cost / \
         (1 + np.exp(self.cost_scaling_factor * (d - self.inflation_radius_w)))

    # -------------------------------------------------
    # UTILITIES
    # -------------------------------------------------
    
    def compute_orientation(self, robot_pos_w, target_point_w):
        """Computes robot orientation given the map robot and target position"""
        x0, y0 = robot_pos_w
        x1, y1 = target_point_w
        angle = math.atan2(y1 - y0, x1 - x0)
        return angle

    def is_free_space(self, pos_w):
        return self.sdf(pos_w) > self.robot_radius_w
    
    def is_visible(self, eye, obj, max_range_w):
        # --- 1. range check ---
        dist_total = compute_dist(eye, obj)
        if dist_total > max_range_w:
            return False

        # --- 2. building vision line ---
        x0, y0 = self.world_to_map(*eye)
        x1, y1 = self.world_to_map(*obj)
        vision_line = bresenham(x0, y0, x1, y1)

        # --- 3. checking visibility ---
        step = self.resolution
        for x, y in vision_line[:-1]:  # excluding the last point (the object itself)
            # --- a. skipping points too close to the eye (to avoid self-collision) ---
            if np.sqrt((x - x0)**2 + (y - y0)**2) < self.robot_radius:
                continue
            
            # --- b. checking map bounds ---
            if not self.is_in_bounds((x, y)):
                return False
            
            # --- c. checking if it is an obstacle or if it is too close ---
            if self.obstacle_map[x, y] >= 1:
                return False
            
            xw, yw = self.map_to_world(x, y)
            if self.sdf((xw, yw)) < step:
                return False

        return True
    
    def is_task_visible(self,tname):
        for objname in self.objects_identified:
            if tname in objname:
                return True
        return False

    def gradient(self, pos_w):
        if self.grad_x is None or self.grad_y is None:
            return (0.0, 0.0)

        mx, my = self.world_to_map(*pos_w)

        ix = int(math.floor(mx))
        iy = int(math.floor(my))

        if not self.is_in_bounds((ix, iy)):
            return (0.0, 0.0)

        return self.grad_x[ix, iy], self.grad_y[ix, iy]
    
    # -------------------------------------------------
    # VISUALIZATION
    # -------------------------------------------------

    def visualize(self, robot=None, path=None, tasks=None, memory_map=None, step=None):

        # Throttle
        if step is not None and step % 10 != 0:
            return

        # -------------------------------------------------
        # Build COST MAP from SDF (vectorized)
        # -------------------------------------------------
        d = self.sdf_map
        viz_cost = np.zeros_like(d)

        # obstacle region
        viz_cost[d <= self.robot_radius_w] = self.max_cost

        # inflation region
        mask = (d > self.robot_radius_w) & (d <= self.inflation_radius_w)
        viz_cost[mask] = self.max_cost / (1 + np.exp(self.cost_scaling_factor * (d[mask] - self.inflation_radius_w)))

        # transpose for visualization
        viz_cost = viz_cost.T
        viz_obstacles = self.obstacle_map.T
        viz_memory = memory_map.T if memory_map is not None else None

        # -------------------------------------------------
        # Initialize figure
        # -------------------------------------------------

        if not hasattr(self, "_fig") or self._fig is None:  

            self._fig, self._ax = plt.subplots(figsize=(6, 6))

            # obstacle overlay (strong)
            self._obs_overlay = self._ax.imshow(
                np.zeros_like(viz_cost),
                cmap='gray',
                vmin=0, vmax=1,
                origin='lower',
                alpha=1.0,
                zorder=1
            )

            # costmap (inflation visualization)
            self._im = self._ax.imshow(
                np.zeros_like(viz_cost),
                cmap='inferno',
                origin='lower',
                vmin=0,
                vmax=self.max_cost,
                alpha=0.5,
                zorder=2
            )


            # memory overlay
            self._memory_overlay = self._ax.imshow(
                np.zeros_like(viz_cost),
                cmap='gray',
                origin='lower',
                vmin=0, vmax=1,
                alpha=0.3,
                zorder=3
            )

            # path
            self._path_line, = self._ax.plot([], [], 'r.-', label="Path", zorder=5)

            # tasks
            self._task_markers, = self._ax.plot([], [], 'bs', label="Tasks", zorder=4)

            # robot
            self._robot_patch = None
            self._visibility_circle = None

            self._ax.set_title("SDF Inflation Map")
            self._ax.legend()

            self._colorbar = self._fig.colorbar(self._im, ax=self._ax)

        # -------------------------------------------------
        # Update layers
        # -------------------------------------------------
        if self._im:
            self._im.set_data(viz_cost)

        if self._obs_overlay:
            self._obs_overlay.set_data(viz_obstacles)

        if viz_memory is not None:
            visited_mask = (viz_memory > 0)
            self._memory_overlay.set_data(visited_mask.astype(float))

        # -------------------------------------------------
        # Robot
        # -------------------------------------------------

        if robot:

            mx, my = self.world_to_map(robot['pos'][0], robot['pos'][1])

            radius_px = robot['radius'] / self.resolution
            vis_px = robot['visibility_radius'] / self.resolution

            # visibility circle
            if self._visibility_circle is None:
                self._visibility_circle = ptc.Circle(
                    (mx, my),
                    vis_px,
                    fill=False,
                    linestyle='--',
                    color='cyan',
                    linewidth=1.5,
                    zorder=5
                )
                if self._ax:
                    self._ax.add_patch(self._visibility_circle)
            else:
                self._visibility_circle.center = (mx, my)

            # robot body
            if self._robot_patch is None:
                self._robot_patch = ptc.Circle(
                    (mx, my),
                    radius_px,
                    color='green',
                    zorder=6
                )
                if self._ax:
                    self._ax.add_patch(self._robot_patch)
            else:
                self._robot_patch.center = (mx, my)


            # -------------------------------------------------
            # Path
            # -------------------------------------------------
            if path:
                transl_path = [(mx,my)]
                for p in path:
                    transl_path.append(
                        (
                            int(math.floor(p[0] / self.resolution)),
                            int(math.floor(p[1] / self.resolution))
                        )
                    )

                px, py = zip(*transl_path)
                self._path_line.set_data(px, py)

            else:
                self._path_line.set_data([], [])

        # -------------------------------------------------
        # Tasks
        # -------------------------------------------------

        if tasks:

            transl_tasks = [
                (
                    int(math.floor(tpos[0] / self.resolution)),
                    int(math.floor(tpos[1] / self.resolution))
                )
                for tpos in tasks.values()
            ]

            tx, ty = zip(*transl_tasks)
            self._task_markers.set_data(tx, ty)

        else:
            self._task_markers.set_data([], [])

        # -------------------------------------------------
        # Redraw
        # -------------------------------------------------

        self._fig.canvas.draw()
        self._fig.canvas.flush_events()