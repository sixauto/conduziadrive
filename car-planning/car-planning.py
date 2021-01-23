import pygame
import pygame_menu
import math
from queue import PriorityQueue

from modules.node import Node 
from modules.graphic_util import *

from algorithms.astar import *
from algorithms.dijkstras import *


WIDTH = 1920
HEIGHT = 1080
ROWS = 48   ### ROW / WIDTH should be an integer (48, 60, 64, 80, 96, 120, 128, 160)
WIN = pygame.display.set_mode((WIDTH, HEIGHT))
SELECTED_ALGORITHM = "dijkstras"
selected_environment = "city"
pygame.display.set_caption("Car planning")

def menu(win):
	pygame.init()
	menu = pygame_menu.Menu(HEIGHT, WIDTH, 'ConduzIA', theme=pygame_menu.themes.THEME_BLUE)
	menu.add_selector('Algorithm :', [('Dijkstra', 1), ('    A*   ', 2)], onchange=set_algorithm)
	menu.add_selector('Environment :', [('Factory', 1), ('   City  ', 2)], onchange=set_environment)
	menu.add_button('Start', start)
	menu.add_button('Quit', pygame_menu.events.EXIT)

	menu.mainloop(win)

def set_algorithm(value, algorithm):
	global SELECTED_ALGORITHM

	if(algorithm == 1):
		SELECTED_ALGORITHM = "dijkstras"
	elif(algorithm == 2):
		SELECTED_ALGORITHM = "astar"

def set_environment(value, environment):
    selected_environment = environment

def start():
	main(WIN)


def main(win):
	print("early " + SELECTED_ALGORITHM)

	grid = make_grid(ROWS, WIDTH)

	start = None
	end = None

	run = True
	while run:
		draw(win, grid, ROWS, WIDTH)
		for event in pygame.event.get():
			if event.type == pygame.QUIT:
				run = False

			if pygame.mouse.get_pressed()[0]: # LEFT
				pos = pygame.mouse.get_pos()
				row, col = get_clicked_pos(pos, ROWS, WIDTH)
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
				row, col = get_clicked_pos(pos, ROWS, WIDTH)
				node = grid[row][col]
				node.reset()
				if node == start:
					start = None
				elif node == end:
					end = None

			if event.type == pygame.KEYDOWN:
				if event.key == pygame.K_SPACE and start and end:
					clean_grid_except_start_end_barriers(grid)
					draw(win, grid, ROWS, WIDTH)
					for row in grid:
						for node in row:
							node.update_neighbors(grid)

					timeBefore = pygame.time.get_ticks()/1000
					print("you selected " + SELECTED_ALGORITHM)
					if(SELECTED_ALGORITHM == "astar"):
						aStarWithVisualizer(lambda: draw(win, grid, ROWS, WIDTH), grid, start, end)
					elif (SELECTED_ALGORITHM == "dijkstras"):
						dijkstrasWithVisualizer(lambda: draw(win, grid, ROWS, WIDTH), grid, start, end)
					timeAfter = pygame.time.get_ticks()/1000
					show_elapsed_time(win, timeBefore, timeAfter)
					

				if event.key == pygame.K_z and start and end:
					for row in grid:
						for node in row:
							node.update_neighbors(grid)
					timeBefore = pygame.time.get_ticks()/1000
					if(SELECTED_ALGORITHM == "astar"):
						aStar(lambda: draw(win, grid, ROWS, WIDTH), grid, start, end)
					elif (SELECTED_ALGORITHM == "dijkstras"):
						dijkstrasWithVisualizer(lambda: draw(win, grid, ROWS, WIDTH), grid, start, end)
					timeAfter = pygame.time.get_ticks()/1000
					show_elapsed_time(win, timeBefore, timeAfter)

				if event.key == pygame.K_c:
					start = None
					end = None
					grid = make_grid(ROWS, WIDTH)

				if event.key == pygame.K_m:
					menu(WIN)

	pygame.quit()


menu(WIN)
