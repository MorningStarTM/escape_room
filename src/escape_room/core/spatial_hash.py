import pygame
from collections import defaultdict

class SpatialHash:
    def __init__(self, cell_size=64):
        self.cell = int(cell_size)
        self.buckets = defaultdict(list)

    def _cells_for_rect(self, r: pygame.Rect):
        x0 = r.left // self.cell
        x1 = r.right // self.cell
        y0 = r.top // self.cell
        y1 = r.bottom // self.cell
        for cy in range(y0, y1 + 1):
            for cx in range(x0, x1 + 1):
                yield (cx, cy)

    def build(self, rects):
        self.buckets.clear()
        for r in rects:
            for c in self._cells_for_rect(r):
                self.buckets[c].append(r)

    def query_rect(self, r: pygame.Rect):
        out = []
        seen = set()
        for c in self._cells_for_rect(r):
            for obj in self.buckets.get(c, []):
                oid = id(obj)
                if oid not in seen:
                    seen.add(oid)
                    out.append(obj)
        return out
