import json
import pygame
from pathlib import Path
import xml.etree.ElementTree as ET

# Tiled flip flags (stored in the high bits of the gid)
FLIP_H = 0x80000000
FLIP_V = 0x40000000
FLIP_D = 0x20000000
GID_MASK = ~(FLIP_H | FLIP_V | FLIP_D)


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_tsx(tsx_path: Path):
    """
    Minimal TSX parser for common single-image tilesets.
    Returns dict: {image, columns, tilecount, tilewidth, tileheight}
    """
    tree = ET.parse(str(tsx_path))
    root = tree.getroot()  # <tileset>

    ts = {
        "tilewidth": int(root.attrib["tilewidth"]),
        "tileheight": int(root.attrib["tileheight"]),
        "tilecount": int(root.attrib.get("tilecount", "0")),
        "columns": int(root.attrib.get("columns", "0")),
    }

    img = root.find("image")
    if img is None or "source" not in img.attrib:
        raise ValueError(f"TSX has no <image source=...>: {tsx_path}")

    ts["image"] = img.attrib["source"]
    return ts


class Tileset:
    def __init__(self, firstgid: int, tsx_path: Path):
        self.firstgid = firstgid
        self.tsx_path = tsx_path

        ts = load_tsx(tsx_path)
        self.columns = ts["columns"]
        self.tilecount = ts["tilecount"]
        self.tile_w = ts["tilewidth"]
        self.tile_h = ts["tileheight"]

        img_path = (tsx_path.parent / ts["image"]).resolve()
        self.image = pygame.image.load(str(img_path)).convert_alpha()

        self.lastgid = self.firstgid + self.tilecount - 1
        self.cache = {}  # key: (gid, fh, fv, fd) -> surface

    def contains_gid(self, gid: int) -> bool:
        return self.firstgid <= gid <= self.lastgid

    def get_tile_surface(self, raw_gid: int):
        """
        raw_gid is the number from layer data (may include flip flags).
        Returns a pygame.Surface for the tile.
        """
        if raw_gid == 0:
            return None

        fh = bool(raw_gid & FLIP_H)
        fv = bool(raw_gid & FLIP_V)
        fd = bool(raw_gid & FLIP_D)

        gid = raw_gid & GID_MASK  # remove flags
        if not self.contains_gid(gid):
            return None

        key = (gid, fh, fv, fd)
        if key in self.cache:
            return self.cache[key]

        local_id = gid - self.firstgid
        col = local_id % self.columns
        row = local_id // self.columns

        rect = pygame.Rect(col * self.tile_w, row * self.tile_h, self.tile_w, self.tile_h)

        surf = pygame.Surface((self.tile_w, self.tile_h), pygame.SRCALPHA)
        surf.blit(self.image, (0, 0), rect)

        # Apply flips/diagonal.
        # Note: Diagonal flip in Tiled is not just a simple rotate in all cases,
        # but this approximation works for most orthogonal tiles.
        if fd:
            # swap axes (approx): rotate 90 then flip horizontal
            surf = pygame.transform.rotate(surf, -90)
            surf = pygame.transform.flip(surf, True, False)

        if fh or fv:
            surf = pygame.transform.flip(surf, fh, fv)

        self.cache[key] = surf
        return surf


class TiledMap:
    def __init__(self, tmj_path: str):
        self.tmj_path = Path(tmj_path)
        self.data = load_json(self.tmj_path)

        self.width = self.data["width"]
        self.height = self.data["height"]
        self.tile_w = self.data["tilewidth"]
        self.tile_h = self.data["tileheight"]

        # layers
        self.layers = [ly for ly in self.data["layers"] if ly["type"] == "tilelayer"]

        # tilesets (multiple)
        self.tilesets = []
        for ts_entry in self.data["tilesets"]:
            firstgid = ts_entry["firstgid"]
            src = ts_entry["source"]  # e.g. "escape.tsx", "side door.tsx"
            tsx_path = (self.tmj_path.parent / src).resolve()
            self.tilesets.append(Tileset(firstgid, tsx_path))

        # sort by firstgid so we can pick correct tileset
        self.tilesets.sort(key=lambda t: t.firstgid)

        # quick sanity: your map uses 16x16 tiles, tilesets should match
        # (not required, but helpful)
        # If they differ, you can still draw, but cropping math changes.
        # We'll assume 16x16 like your tmj.
        # (If mismatch happens, tell me — we’ll adapt.)

    def _find_tileset(self, gid_without_flags: int):
        # choose the tileset with the highest firstgid <= gid
        chosen = None
        for ts in self.tilesets:
            if ts.firstgid <= gid_without_flags:
                chosen = ts
            else:
                break
        return chosen

    def draw(self, screen: pygame.Surface, camera_x=0, camera_y=0):
        for layer in self.layers:
            data = layer["data"]  # your tmj shows it's already a list ✅

            for y in range(self.height):
                for x in range(self.width):
                    raw_gid = data[y * self.width + x]
                    if raw_gid == 0:
                        continue

                    gid = raw_gid & GID_MASK
                    ts = self._find_tileset(gid)
                    if ts is None:
                        continue

                    tile = ts.get_tile_surface(raw_gid)
                    if tile:
                        screen.blit(tile, (x * self.tile_w - camera_x,
                                           y * self.tile_h - camera_y))




def blit_fit(screen, world_surf):
    ww, wh = world_surf.get_size()
    sw, sh = screen.get_size()

    scale = min(sw / ww, sh / wh)  # fit inside screen, keep aspect ratio
    new_w = int(ww * scale)
    new_h = int(wh * scale)

    scaled = pygame.transform.scale(world_surf, (new_w, new_h))  # crisp
    x = (sw - new_w) // 2
    y = (sh - new_h) // 2
    screen.fill((0, 0, 0))
    screen.blit(scaled, (x, y))

"""def main():
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
    clock = pygame.time.Clock()

    tmap = TiledMap("finalized_escape.tmj")

    world_w = tmap.width * tmap.tile_w
    world_h = tmap.height * tmap.tile_h
    world = pygame.Surface((world_w, world_h), pygame.SRCALPHA)

    cam_x = cam_y = 0  # keep if you want scrolling later

    running = True
    while running:
        clock.tick(60)
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                running = False

        # draw map onto world surface (native resolution)
        world.fill((20, 20, 20))
        tmap.draw(world, cam_x, cam_y)

        # fit world into 720x640
        blit_fit(screen, world)
        pygame.display.flip()

    pygame.quit()


if __name__ == "__main__":
    main()"""