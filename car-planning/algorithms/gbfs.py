import math
from queue import PriorityQueue

import pygame

from modules.graphic_util import *


def GBFSWithVisualizer(draw, grid, start, end):
    count = 0
    open_set = PriorityQueue()
    open_set.put((0, 0, start))
    came_from = {}
    g_score = {node: float("inf") for row in grid for node in row}
    g_score[start] = 0
    f_score = {node: float("inf") for row in grid for node in row}
    f_score[start] = manhattan(start.get_pos(), end.get_pos())

    visited_set = {start}

    while not open_set.empty():
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()

        current = open_set.get()[2]

        if current == end:
            reconstruct_path(came_from, end, draw, start)
            end.make_end()
            return True

        best_distance = float("inf")
        best_neighbour = None
        for neighbor in current.neighbors:
            manhattan_distance = manhattan(neighbor.get_pos(), end.get_pos())
            if neighbor not in visited_set:
                came_from[neighbor] = current
                count += 1
                open_set.put((manhattan_distance, count, neighbor))
                visited_set.add(neighbor)
                neighbor.make_open()

        draw()

        if current != start:
            current.make_closed()

    return False


def manhattan(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    return abs(x1 - x2) + abs(y1 - y2)
