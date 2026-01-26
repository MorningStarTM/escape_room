import pygame
from src.escape_room.constants import *



class SpriteSheet:
    def __init__(self, path):
        self.sheet = pygame.image.load(path).convert_alpha()

    def _crop(self, col, row):
        rect = pygame.Rect(col * FRAME_W, row * FRAME_H, FRAME_W, FRAME_H)
        img = pygame.Surface((FRAME_W, FRAME_H), pygame.SRCALPHA)
        img.blit(self.sheet, (0, 0), rect)
        return img

    def get_full_frame(self, col, top_row):
        """Combine upper+lower (48x48 + 48x48) into one 48x96 frame."""
        top = self._crop(col, top_row)
        bottom = self._crop(col, top_row + 1)

        full = pygame.Surface((FRAME_W, FRAME_H * 2), pygame.SRCALPHA)
        full.blit(top, (0, 0))
        full.blit(bottom, (0, FRAME_H))
        return full

    def get_full_row_frames(self, top_row, start_col=0, end_col=8):
        frames = []
        for c in range(start_col, end_col):
            frames.append(self.get_full_frame(c, top_row))
        return frames