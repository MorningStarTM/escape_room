# src/escape_room/core/rays.py
import math
import pygame


def _clamp_angle_pi(a: float) -> float:
    """Clamp angle to [-pi, +pi] for stability."""
    while a <= -math.pi:
        a += 2.0 * math.pi
    while a > math.pi:
        a -= 2.0 * math.pi
    return a


class Ray:
    """
    A single ray cast in the world.
    - origin: Vector2
    - angle: radians
    - max_dist: pixels
    """

    def __init__(self, max_dist: float = 300.0):
        self.max_dist = float(max_dist)
        self.origin = pygame.Vector2(0, 0)
        self.angle = 0.0
        self.hit = False
        self.hit_point = pygame.Vector2(0, 0)
        self.distance = self.max_dist

    def set(self, origin: pygame.Vector2, angle_rad: float):
        self.origin = pygame.Vector2(origin)
        self.angle = _clamp_angle_pi(float(angle_rad))

    def cast_to_rects(self, obstacles):
        """
        Cast ray against a list/iterable of pygame.Rect.
        Returns:
            (hit: bool, hit_point: Vector2, distance: float)
        """
        dx = math.cos(self.angle)
        dy = math.sin(self.angle)

        # If max_dist is large, don't do per-pixel stepping too slow.
        # We'll do stepping with step_size pixels and also use rect.clipline for exact intersection.
        # Approach:
        # 1) Take the full ray segment end point
        # 2) For each rect, use rect.clipline to get intersection segment
        # 3) Choose closest intersection
        end = pygame.Vector2(
            self.origin.x + dx * self.max_dist,
            self.origin.y + dy * self.max_dist,
        )

        best_dist = self.max_dist
        best_pt = pygame.Vector2(end)
        hit_any = False

        ox, oy = float(self.origin.x), float(self.origin.y)
        ex, ey = float(end.x), float(end.y)

        for r in obstacles:
            # clipline returns () if no intersection, or ((x1,y1),(x2,y2)) for clipped segment
            clipped = r.clipline(ox, oy, ex, ey)
            if clipped:
                (ix1, iy1), (ix2, iy2) = clipped
                # Choose the closer intersection point to origin
                d1 = (ix1 - ox) * (ix1 - ox) + (iy1 - oy) * (iy1 - oy)
                d2 = (ix2 - ox) * (ix2 - ox) + (iy2 - oy) * (iy2 - oy)
                if d2 < d1:
                    ix, iy = ix2, iy2
                    dsq = d2
                else:
                    ix, iy = ix1, iy1
                    dsq = d1

                dist = math.sqrt(dsq)
                if dist < best_dist:
                    best_dist = dist
                    best_pt.update(ix, iy)
                    hit_any = True

        self.hit = hit_any
        self.hit_point = best_pt
        self.distance = best_dist
        return self.hit, self.hit_point, self.distance


class RaySensor180:
    """
    180-degree vision sensor:
    - fov_deg fixed at 180 by default (you can change if needed)
    - n_rays: number of rays across the FOV
    - max_dist: ray length in pixels

    Usage:
        sensor = RaySensor180(fov_deg=180, n_rays=31, max_dist=300)
        sensor.update(origin, facing_angle_rad, obstacles_rects)
        sensor.draw(screen)  # debug
    """

    def __init__(self, fov_deg: float = 180.0, n_rays: int = 31, max_dist: float = 300.0):
        self.fov_deg = float(fov_deg)
        self.n_rays = int(n_rays)
        self.max_dist = float(max_dist)

        if self.n_rays < 1:
            self.n_rays = 1

        # ✅ define fov_rad first
        self.fov_rad = math.radians(self.fov_deg)
        half = 0.5 * self.fov_rad

        # ✅ offsets across [-half, +half]
        if self.n_rays == 1:
            self.offsets = [0.0]
        else:
            step = self.fov_rad / float(self.n_rays - 1)
            self.offsets = [(-half + i * step) for i in range(self.n_rays)]

        self.rays = [Ray(max_dist=self.max_dist) for _ in range(self.n_rays)]

        # outputs (easy to feed into RL later)
        self.distances = [self.max_dist for _ in range(self.n_rays)]
        self.hit_points = [pygame.Vector2(0, 0) for _ in range(self.n_rays)]
        self.hits = [False for _ in range(self.n_rays)]


    
    def update(self, origin: pygame.Vector2, facing_angle_rad: float, obstacles):
        """
        origin: Vector2 (sensor position)
        facing_angle_rad: radians (where the player is looking)
        obstacles: list[pygame.Rect]
        """
        ox = float(origin.x)
        oy = float(origin.y)
        base = float(facing_angle_rad)

        # loop rays using precomputed offsets
        for i, off in enumerate(self.offsets):
            a = base + off
            ray = self.rays[i]

            # avoid creating new Vector2 each ray
            ray.origin.x = ox
            ray.origin.y = oy
            ray.angle = _clamp_angle_pi(a)

            hit, pt, dist = ray.cast_to_rects(obstacles)

            self.hits[i] = hit
            # pt might already be Vector2; if it's tuple, assign without new Vector2
            if isinstance(pt, pygame.Vector2):
                self.hit_points[i].x = pt.x
                self.hit_points[i].y = pt.y
            else:
                self.hit_points[i].x = pt[0]
                self.hit_points[i].y = pt[1]

            self.distances[i] = float(dist)

        return self.distances


    def draw(self, surface, color=(255, 255, 0), hit_color=(255, 80, 80)):
        """
        Debug draw: rays + hit points.
        (Colors are defaults; you can change later.)
        """
        for i, ray in enumerate(self.rays):
            a = ray.origin
            b = ray.hit_point if ray.hit else pygame.Vector2(
                ray.origin.x + math.cos(ray.angle) * ray.max_dist,
                ray.origin.y + math.sin(ray.angle) * ray.max_dist,
            )

            pygame.draw.line(surface, hit_color if ray.hit else color, a, b, 1)

            if ray.hit:
                pygame.draw.circle(surface, hit_color, (int(b.x), int(b.y)), 3)
        