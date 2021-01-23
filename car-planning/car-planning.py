import pygame
import math
from queue import PriorityQueue

from modules.node import Node 
from modules.graphic_util import *

from algorithms.astar import *

WIDTH = 1920
HEIGHT = 1080
ROWS = 48   ### ROW / WIDTH should be an integer (48, 60, 64, 80, 96, 120, 128, 160)
WIN = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Car planning")


def main(win, width):
	grid = make_grid(ROWS, width)

	start = None
	end = None

	run = True
	while run:
		draw(win, grid, ROWS, width)
		for event in pygame.event.get():
			if event.type == pygame.QUIT:
				run = False

			if pygame.mouse.get_pressed()[0]: # LEFT
				pos = pygame.mouse.get_pos()
				row, col = get_clicked_pos(pos, ROWS, width)
				node = grid[row][col]
				if not start and node != end:
					start = node
					start.make_start()

				elif not end and node != start:
					end = node
					end.make_end()

				elif node != end and node != start:
					node.make_barrier()

			elif pygame.mouse.get_pressed()[2]: # RIGHT
				pos = pygame.mouse.get_pos()
				row, col = get_clicked_pos(pos, ROWS, width)
				node = grid[row][col]
				node.reset()
				if node == start:
					start = None
				elif node == end:
					end = None

			if event.type == pygame.KEYDOWN:
				if event.key == pygame.K_SPACE and start and end:
					clean_grid_except_start_end_barriers(grid)
					draw(win, grid, ROWS, width)
					for row in grid:
						for node in row:
							node.update_neighbors(grid)

					aStarWithVisualizer(lambda: draw(win, grid, ROWS, width), grid, start, end)

				if event.key == pygame.K_z and start and end:
					for row in grid:
						for node in row:
							node.update_neighbors(grid)

					aStar(lambda: draw(win, grid, ROWS, width), grid, start, end)

				if event.key == pygame.K_c:
					start = None
					end = None
					grid = make_grid(ROWS, width)

	pygame.quit()

main(WIN, WIDTH)