import json
import pygame
from pathlib import Path
import xml.etree.ElementTree as ET

# Tiled flip flags (stored in the high bits of the gid)
FLIP_H = 0x80000000
FLIP_V = 0x40000000
FLIP_D = 0x20000000
GID_MASK = ~(FLIP_H | FLIP_V | FLIP_D)



def _resolve_image_path(base_dir: Path, img_src_raw: str) -> Path:
    img_src = Path(img_src_raw)

    candidates = [
        (base_dir / img_src),
        (base_dir / img_src.name),  # basename fallback
    ]

    if "\\" in img_src_raw:
        img_src2 = Path(img_src_raw.replace("\\", "/"))
        candidates.insert(1, base_dir / img_src2)
        candidates.append(base_dir / img_src2.name)

    for c in candidates:
        c = c.resolve()
        if c.exists():
            return c

    raise FileNotFoundError(
        "Tileset image not found.\n"
        f"base_dir: {base_dir}\n"
        f"image source: {img_src_raw}\n"
        "tried:\n  " + "\n  ".join(str(x.resolve()) for x in candidates)
    )


def load_obstacle_rects_from_tmx(tmx_path: str, layer_name: str = "obstacle"):
    """
    Reads a TMX and returns pygame.Rect list from an Object Layer (objectgroup).
    This uses the object's x,y,width,height (rect objects).
    """
    tmx_path = str(Path(tmx_path))
    tree = ET.parse(tmx_path)
    root = tree.getroot()

    rects = []
    for obj_group in root.findall("objectgroup"):
        if obj_group.attrib.get("name") != layer_name:
            continue

        for obj in obj_group.findall("object"):
            x = float(obj.attrib.get("x", 0))
            y = float(obj.attrib.get("y", 0))
            w = float(obj.attrib.get("width", 0))
            h = float(obj.attrib.get("height", 0))

            # ignore non-rect objects
            if w <= 0 or h <= 0:
                continue

            rects.append(pygame.Rect(int(x), int(y), int(w), int(h)))

    return rects


def _obj_to_rect(o: dict) -> pygame.Rect:
    # Tiled object: x,y is top-left in pixels
    x = int(o.get("x", 0))
    y = int(o.get("y", 0))
    w = int(o.get("width", 0))
    h = int(o.get("height", 0))
    return pygame.Rect(x, y, w, h)


def extract_collision_from_tmj(tmj: dict):
    """
    Reads Tiled object layers:
      - layer name "Walls"  => solid rects
      - layer name "Doors"  => door rects (can be opened/locked)
    Returns:
      wall_rects: list[pygame.Rect]
      doors: list[dict]  { "rect": Rect, "name": str, "open": bool, "locked": bool }
    """
    wall_rects = []
    doors = []

    for layer in tmj.get("layers", []):
        if layer.get("type") != "objectgroup":
            continue

        lname = (layer.get("name") or "").strip().lower()
        objects = layer.get("objects", [])

        if lname == "walls":
            for o in objects:
                wall_rects.append(_obj_to_rect(o))

        elif lname == "doors":
            for o in objects:
                rect = _obj_to_rect(o)
                props = {p["name"]: p.get("value") for p in o.get("properties", [])} if o.get("properties") else {}
                doors.append({
                    "rect": rect,
                    "name": str(o.get("name", "")),
                    "open": bool(props.get("open", False)),
                    "locked": bool(props.get("locked", False)),
                })

    return wall_rects, doors



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



class TilesetEmbedded:
    def __init__(self, firstgid: int, ts_entry: dict, tmj_dir: Path):
        self.firstgid = int(firstgid)

        self.columns = int(ts_entry.get("columns", 0))
        self.tilecount = int(ts_entry.get("tilecount", 0))
        self.tile_w = int(ts_entry.get("tilewidth", 0))
        self.tile_h = int(ts_entry.get("tileheight", 0))

        img_src_raw = ts_entry.get("image")
        if not img_src_raw:
            raise ValueError(f"Embedded tileset has no 'image': firstgid={firstgid}")

        img_path = _resolve_image_path(tmj_dir, img_src_raw)
        self.image = pygame.image.load(str(img_path)).convert_alpha()

        self.lastgid = self.firstgid + self.tilecount - 1
        self.cache = {}

    def contains_gid(self, gid: int) -> bool:
        return self.firstgid <= gid <= self.lastgid

    def get_tile_surface(self, raw_gid: int):
        # same logic as your Tileset.get_tile_surface
        if raw_gid == 0:
            return None

        fh = bool(raw_gid & FLIP_H)
        fv = bool(raw_gid & FLIP_V)
        fd = bool(raw_gid & FLIP_D)

        gid = raw_gid & GID_MASK
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

        if fd:
            surf = pygame.transform.rotate(surf, -90)
            surf = pygame.transform.flip(surf, True, False)
        if fh or fv:
            surf = pygame.transform.flip(surf, fh, fv)

        self.cache[key] = surf
        return surf



class Tileset:
    def __init__(self, firstgid: int, tsx_path: Path):
        self.firstgid = firstgid
        self.tsx_path = Path(tsx_path)

        ts = load_tsx(self.tsx_path)
        self.columns = ts["columns"]
        self.tilecount = ts["tilecount"]
        self.tile_w = ts["tilewidth"]
        self.tile_h = ts["tileheight"]

        # -------- FIX: robust image path resolution --------
        img_src_raw = ts["image"]  # e.g. "Tileset.png" or "src/escape_room/assets/images/Tileset.png"
        img_src = Path(img_src_raw)

        # If TSX already contains an absolute path, just use it
        if img_src.is_absolute() and img_src.exists():
            img_path = img_src
        else:
            img_candidates = []

            # 1) relative to the TSX folder (normal tiled behavior)
            img_candidates.append(self.tsx_path.parent / img_src)
            img_candidates.append(self.tsx_path.parent / img_src.name)

            # 2) also try walking UP the directory tree (so "src/escape_room/..." works)
            #    This will find: E:\Projects\escape_room\src\escape_room\assets\images\Tileset.png
            for parent in self.tsx_path.parents:
                img_candidates.append(parent / img_src)
                img_candidates.append(parent / img_src.name)

            # 3) handle backslashes stored in TSX (windows paths)
            if "\\" in img_src_raw:
                img_src2 = Path(img_src_raw.replace("\\", "/"))
                for parent in self.tsx_path.parents:
                    img_candidates.append(parent / img_src2)
                    img_candidates.append(parent / img_src2.name)

            img_path = None
            for c in img_candidates:
                c = c.resolve()
                if c.exists():
                    img_path = c
                    break

            if img_path is None:
                raise FileNotFoundError(
                    "Tileset image not found.\n"
                    f"tsx: {self.tsx_path}\n"
                    f"image source in tsx: {img_src_raw}\n"
                    "tried:\n  " + "\n  ".join(str(x.resolve()) for x in img_candidates[:12]) +
                    ("\n  ... (more tried)" if len(img_candidates) > 12 else "")
                )

        self.image = pygame.image.load(str(img_path)).convert_alpha()
        # -----------------------------------------------

        # -----------------------------------------------

        self.lastgid = self.firstgid + self.tilecount - 1
        self.cache = {}  # key: (gid, fh, fv, fd) -> surface

    def contains_gid(self, gid: int) -> bool:
        return self.firstgid <= gid <= self.lastgid

    def get_tile_surface(self, raw_gid: int):
        if raw_gid == 0:
            return None

        fh = bool(raw_gid & FLIP_H)
        fv = bool(raw_gid & FLIP_V)
        fd = bool(raw_gid & FLIP_D)

        gid = raw_gid & GID_MASK
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

        if fd:
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
        self._static_world = None
        self._static_dirty = True

        self.layers = [ly for ly in self.data["layers"] if ly["type"] == "tilelayer"]
        self.tile_layers = [ly for ly in self.data["layers"] if ly["type"] == "tilelayer"]
        self.object_layers = [ly for ly in self.data["layers"] if ly["type"] == "objectgroup"]

        # -------- FIX: robust TSX path resolution --------
        self.tilesets = []
        tmj_dir = self.tmj_path.parent

        for ts_entry in self.data.get("tilesets", []):
            firstgid = ts_entry["firstgid"]

            # Case A: external TSX reference
            if "source" in ts_entry:
                src_raw = ts_entry["source"]
                src = Path(src_raw)

                candidates = [
                    (tmj_dir / src),
                    (tmj_dir / src.name),
                ]

                if "\\" in src_raw:
                    src2 = Path(src_raw.replace("\\", "/"))
                    candidates.insert(1, tmj_dir / src2)
                    candidates.append(tmj_dir / src2.name)

                tsx_path = None
                for c in candidates:
                    c = c.resolve()
                    if c.exists():
                        tsx_path = c
                        break

                if tsx_path is None:
                    raise FileNotFoundError(
                        "Tileset TSX not found.\n"
                        f"tmj: {self.tmj_path}\n"
                        f"source in tmj: {src_raw}\n"
                        "tried:\n  " + "\n  ".join(str(x.resolve()) for x in candidates)
                    )

                self.tilesets.append(Tileset(firstgid, tsx_path))

            # Case B: embedded tileset inside TMJ (no 'source', has 'image')
            elif "image" in ts_entry:
                self.tilesets.append(TilesetEmbedded(firstgid, ts_entry, tmj_dir))

            else:
                raise ValueError(f"Unknown tileset format in TMJ: {ts_entry}")

        self.tilesets.sort(key=lambda t: t.firstgid)



        # quick sanity: your map uses 16x16 tiles, tilesets should match
        # (not required, but helpful)
        # If they differ, you can still draw, but cropping math changes.
        # We'll assume 16x16 like your tmj.
        # (If mismatch happens, tell me — we’ll adapt.)

    def get_tile_layer_by_name(self, name: str):
        for ly in self.tile_layers:
            if ly.get("name") == name:
                return ly
        return None

    def get_object_layer_by_name(self, name: str):
        for ly in self.object_layers:
            if ly.get("name") == name:
                return ly
        return None

    def _find_tileset(self, gid_without_flags: int):
        # choose the tileset with the highest firstgid <= gid
        chosen = None
        for ts in self.tilesets:
            if ts.firstgid <= gid_without_flags:
                chosen = ts
            else:
                break
        return chosen

    def draw(self, screen: pygame.Surface, camera_x=0, camera_y=0, skip_cells_by_layer=None):
        """
        skip_cells_by_layer: dict[layer_name] -> set[(tx, ty)]
        """
        skip_cells_by_layer = skip_cells_by_layer or {}

        for layer in self.tile_layers:
            data = layer["data"]
            lname = layer.get("name", "")

            skip = skip_cells_by_layer.get(lname, set())

            for y in range(self.height):
                for x in range(self.width):
                    if (x, y) in skip:
                        continue

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



    def build_static_world(self):
        """Render the full map once into a surface."""
        world_w = self.width * self.tile_w
        world_h = self.height * self.tile_h
        surf = pygame.Surface((world_w, world_h), pygame.SRCALPHA)

        # draw all layers ONCE
        for layer in self.layers:
            data = layer["data"]
            for y in range(self.height):
                row_off = y * self.width
                py = y * self.tile_h
                for x in range(self.width):
                    raw_gid = data[row_off + x]
                    if raw_gid == 0:
                        continue
                    gid = raw_gid & GID_MASK
                    ts = self._find_tileset(gid)
                    if ts is None:
                        continue
                    tile = ts.get_tile_surface(raw_gid)
                    if tile:
                        surf.blit(tile, (x * self.tile_w, py))

        self._static_world = surf
        self._static_dirty = False

    def draw_cached(self, screen: pygame.Surface, camera_x=0, camera_y=0):
        """Fast draw: just blit the cached full-map surface."""
        if self._static_world is None or self._static_dirty:
            self.build_static_world()
        screen.blit(self._static_world, (-camera_x, -camera_y))


    def draw_tile_layer(self, screen, layer_name: str, camera_x=0, camera_y=0, skip_cells=None):
        layer = self.get_tile_layer_by_name(layer_name)
        if not layer:
            return

        skip_cells = skip_cells or set()
        data = layer["data"]

        for y in range(self.height):
            for x in range(self.width):
                if (x, y) in skip_cells:
                    continue

                raw_gid = data[y * self.width + x]
                if raw_gid == 0:
                    continue

                gid = raw_gid & GID_MASK
                ts = self._find_tileset(gid)
                if ts is None:
                    continue

                tile = ts.get_tile_surface(raw_gid)
                if tile:
                    screen.blit(tile, (x * self.tile_w - camera_x, y * self.tile_h - camera_y))


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