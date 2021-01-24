from modules.color import Color
from modules.node import Node

import pygame
import pickle

def make_grid(rows, width):
    grid = []
    gap = width // rows
    for i in range(rows):
        grid.append([])
        for j in range(rows):
            node = Node(i, j, gap, rows)
            grid[i].append(node)

    return grid

def draw_grid(win, rows, width):
    gap = width // rows
    for i in range(rows):
        pygame.draw.line(win, Color.GREY, (0, i * gap), (width, i * gap))
        for j in range(rows):
            pygame.draw.line(win, Color.GREY, (j * gap, 0), (j * gap, width))

def draw(win, grid, rows, width):
    win.fill(Color.WHITE)

    for row in grid:
        for node in row:
            node.draw(win)

    draw_grid(win, rows, width)
    pygame.display.update()

def get_clicked_pos(pos, rows, width):
    gap = width // rows
    y, x = pos

    row = y // gap
    col = x // gap

    return row, col

def reconstruct_path(came_from, current, draw, start_node):
    while current in came_from:
        current = came_from[current]
        if(current.get_pos() == start_node.get_pos()): #if we reached the start node we don't want to color it purple, so we ignore it
            continue
        current.make_path()
        draw()

def clean_grid_except_start_end_barriers(grid):
    for row in grid:
        for node in row:
            if(node.is_closed() or node.is_open() or node.is_path()):
                node.reset()
            
def show_elapsed_time(win, time_before, time_after):
    time_string = "It took {:.2f} seconds to find the path".format(time_after - time_before)
    font = pygame.font.SysFont("Impact", 30)
    text = font.render(time_string, True, Color.RED)
    width, height = pygame.display.get_surface().get_size()
    win.blit(text, (width/3, height/2))
    pygame.display.update()
    pygame.time.wait(3000)

def save_grid_to_file(grid):
    for row in grid:
        for node in row:
            if(node.is_closed() or node.is_open() or node.is_path() or node.is_start() or node.is_end()):
                node.reset()

    with open('grid_file', 'wb') as f:  
        pickle.dump(grid, f)

def load_grid_from_file(map_name):
    with open(map_name, 'rb') as f:
        grid = pickle.load(f)
        return grid