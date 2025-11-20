import pygame
import numpy as np
import sys
import math

# -----------------------------
#     ENTRADAS DEL USUARIO
# -----------------------------
N = int(input("Elige el número de partículas: "))
rastro = int(input("Elige la cantidad de rastro: "))
G1 = float(input("Elige la intensidad de la gravedad (se dividirá entre 10): "))
G = G1 / 10
stop_radius = float(input("Elige el radio de captura (se dividirá entre 10): ")) / 10

dispersion_inicial = float(input("Elige la dispersión inicial de las partículas: ")) * 5
min_disp = -dispersion_inicial

cirle_area = int(input("Elige el área del círculo que hace el centro de gravedad: "))

black_hole = int(input("Modo agujero negro (0: no, 1: sí): "))
if black_hole == 1:
    growth_factor = float(input("Cuánto aumenta G por captura (se dividirá entre 50): ")) / 50
    radius_growth = float(input("Cuánto aumenta stop_radius por captura (se dividirá entre 100): ")) / 100
else:
    growth_factor = 0.0
    radius_growth = 0.0

# -----------------------------
#        CONFIGURACIÓN
# -----------------------------
pygame.init()
WIDTH, HEIGHT = 1000, 1000
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Simulador en Pygame")

clock = pygame.time.Clock()

# convertir coords matemáticas a coords de pantalla
def to_screen_coords(pos):
    x = int(WIDTH/2 + pos[0]*20)
    y = int(HEIGHT/2 - pos[1]*20)
    return x, y

# límites para dibujar
max_coord = cirle_area + 20

# -----------------------------
#   INICIALIZAR PARTÍCULAS
# -----------------------------
pos = np.random.uniform(min_disp, dispersion_inicial, (N, 2))
vel = np.zeros((N, 2))

gravity_center = np.array([0.0, 0.0])

dt = 0.3
damping = 0.01
max_speed = 2.0

collision_radius = 0.3
repulsion_strength = 0.1

# rastro
paths = [[] for _ in range(N)]

# capturas
N_captured_pts = 0
captured_positions = []

# -----------------------------
#          BUCLE
# -----------------------------
running = True
frame = 0

while running:

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    frame += 1

    screen.fill((0, 0, 0))

    # mover centro
    gravity_center = np.array([
        cirle_area * math.sin(frame * 0.02),
        cirle_area * math.cos(frame * 0.02)
    ])

    # -------- RASTROS --------
    for i in range(pos.shape[0]):
        paths[i].append(pos[i].copy())
        if len(paths[i]) > rastro:
            paths[i].pop(0)

    # -------- FÍSICA --------
    direction = gravity_center - pos
    dist = np.linalg.norm(direction, axis=1).reshape(-1, 1)
    dist[dist == 0] = 0.001

    acc = (G * direction) / (dist ** 1.5)
    max_acc = 50.0

    acc_norm = np.linalg.norm(acc, axis=1).reshape(-1, 1)
    too_strong = acc_norm > max_acc
    if np.any(too_strong):
        acc[too_strong.flatten()] *= (max_acc / acc_norm[too_strong])

    vel += acc * dt
    vel *= (1.0 - damping)

    speed = np.linalg.norm(vel, axis=1)
    speed_safe = np.where(speed == 0, 1, speed)
    exceed = speed > max_speed
    vel[exceed] *= (max_speed / speed_safe[exceed])[:, None]

    pos += vel * dt

    # repulsión
    M = pos.shape[0]
    if M > 1:
        for i in range(M):
            for j in range(i + 1, M):
                diff = pos[i] - pos[j]
                d = np.linalg.norm(diff)
                if d < collision_radius and d > 0:
                    push = (diff / d) * repulsion_strength
                    vel[i] += push
                    vel[j] -= push

    # -------- CAPTURA --------
    if pos.shape[0] > 0:
        close = np.linalg.norm(pos - gravity_center, axis=1) < stop_radius
    else:
        close = np.array([], dtype=bool)

    if close.any():
        captured_pts = pos[close].copy()

        for p in captured_pts:
            captured_positions.append(p.copy())

        N_captured_pts += captured_pts.shape[0]

        if black_hole == 1:
            G += growth_factor * captured_pts.shape[0]
            stop_radius += radius_growth * captured_pts.shape[0]

        keep_idx = np.where(~close)[0]
        pos = pos[keep_idx]
        vel = vel[keep_idx]
        paths = [paths[i] for i in keep_idx]

    # -------- DIBUJO --------

    # rastro
    for i in range(pos.shape[0]):
        if len(paths[i]) > 1:
            for a in range(len(paths[i]) - 1):
                x1, y1 = to_screen_coords(paths[i][a])
                x2, y2 = to_screen_coords(paths[i][a + 1])
                pygame.draw.line(screen, (100, 100, 255), (x1, y1), (x2, y2), 1)

    # partículas (tamaño fijo pequeño como en numpy)
    for i in range(pos.shape[0]):
        x, y = to_screen_coords(pos[i])
        pygame.draw.circle(screen, (0, 150, 255), (x, y), 3)   # radio = 2

    # centro
    cx, cy = to_screen_coords(gravity_center)
    pygame.draw.circle(screen, (255, 0, 0), (cx, cy), int(stop_radius * 20), 2)

    # marcadores de captura
    for p in captured_positions:
        x, y = to_screen_coords(p)
        pygame.draw.circle(screen, (255, 255, 0), (x, y), 4)

    # si se capturan todas
    if N_captured_pts >= N:
        print("\nTODAS LAS PARTÍCULAS CAPTURADAS:\n")
        for i, p in enumerate(captured_positions, 1):
            print(f"{i}: {p[0]:.6f}, {p[1]:.6f}")
        running = False

    pygame.display.flip()
    clock.tick(60)

pygame.quit()
sys.exit()
