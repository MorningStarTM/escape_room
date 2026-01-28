# src/escape_room/core/doors.py
import pygame


class Door:
    def __init__(self, blocker_rect: pygame.Rect, trigger_rect: pygame.Rect | None = None, tiles_cells=None):
        self.blocker_rect = blocker_rect.copy()
        self.trigger_rect = trigger_rect.copy() if trigger_rect else blocker_rect.inflate(40, 40)
        self.open = False
        self.tiles_cells = tiles_cells or set()

    def can_interact(self, player_rect: pygame.Rect, max_dist_px: int = 50) -> bool:
        dx = player_rect.centerx - self.trigger_rect.centerx
        dy = player_rect.centery - self.trigger_rect.centery
        return (dx * dx + dy * dy) <= (max_dist_px * max_dist_px)

    def toggle(self):
        self.open = not self.open

    def update_auto_close(self, player_rect: pygame.Rect):
        # Close when player is no longer inside trigger area
        if self.open and (not self.trigger_rect.colliderect(player_rect)):
            self.open = False

    def blocker_rects(self):
        return [] if self.open else [self.blocker_rect]


def build_doors_from_tiled(tmap, doors_layer_name="Doors", door_tiles_layer_name="DoorsTiles"):
    """
    Reads Tiled object layer "Doors" (objectgroup).
    Each object rectangle becomes a door blocker (closed), and a trigger zone.
    Also computes which tile-cells in DoorsTiles to hide when open.
    """
    doors = []

    doors_layer = tmap.get_object_layer_by_name(doors_layer_name)
    if not doors_layer:
        print(f"[DOORS] No object layer named '{doors_layer_name}' found.")
        return doors

    has_door_tiles_layer = (tmap.get_tile_layer_by_name(door_tiles_layer_name) is not None)

    for obj in doors_layer.get("objects", []):
        x = int(obj.get("x", 0))
        y = int(obj.get("y", 0))
        w = int(obj.get("width", 0))
        h = int(obj.get("height", 0))

        blocker = pygame.Rect(x, y, w, h)
        trigger = blocker.inflate(60, 60)

        cells = set()
        if has_door_tiles_layer:
            tx0 = blocker.left // tmap.tile_w
            ty0 = blocker.top  // tmap.tile_h
            tx1 = (blocker.right - 1) // tmap.tile_w
            ty1 = (blocker.bottom - 1)// tmap.tile_h
            for ty in range(ty0, ty1 + 1):
                for tx in range(tx0, tx1 + 1):
                    cells.add((tx, ty))

        doors.append(Door(blocker_rect=blocker, trigger_rect=trigger, tiles_cells=cells))

    print(f"[DOORS] Loaded {len(doors)} doors from '{doors_layer_name}'.")
    return doors
