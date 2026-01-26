SCREEN_W, SCREEN_H = 720, 640


FRAME_W, FRAME_H = 48, 48
SCALE = 1
FPS = 60

ANIM_FPS_IDLE = 8
ANIM_FPS_WALK = 10
ANIM_FPS_RUN  = 14

# top-row per direction (because each direction uses 2 rows: top + bottom)
DIR_TOPROW = {
    "down": 0,   # 0(top) + 1(bottom)
    "left": 2,   # 2 + 3
    "right": 4,  # 4 + 5
    "up": 6,     # 6 + 7
}

ANIM_FPS = {
    "idle": 8,
    "walk": 10,
    "run": 14,
}


#map path
map_low = "src/escape_room/assets/maps/finalized_escape.tmj"

#player animation paths
idle_path="src/escape_room/assets/Male/Male_I.png"
walk_path="src/escape_room/assets/Male/Male_W.png"
run_path="src/escape_room/assets/Male/Male_R.png"