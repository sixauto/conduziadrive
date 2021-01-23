import math
from queue import PriorityQueue
import pygame

from modules.graphic_util import *


def dijkstrasWithVisualizer(draw, grid, start, end):
	count = 0
	open_set = PriorityQueue()
	open_set.put((0, count, start))
	came_from = {}
	distance_to_the_start = {node: float("inf") for row in grid for node in row}
	distance_to_the_start[start] = 0

	open_set_hash = {start}

	while not open_set.empty():
		for event in pygame.event.get():
			if event.type == pygame.QUIT:
				pygame.quit()

		current = open_set.get()[2]
		open_set_hash.remove(current)

		if current == end:
			reconstruct_path(came_from, end, draw, start)
			end.make_end()
			return True

		for neighbor in current.neighbors:
			temp_distance_to_the_start = distance_to_the_start[current] + 1

			if temp_distance_to_the_start < distance_to_the_start[neighbor]:
				came_from[neighbor] = current
				distance_to_the_start[neighbor] = temp_distance_to_the_start
				if neighbor not in open_set_hash:
					count += 1
					open_set.put((distance_to_the_start[neighbor], count, neighbor))
					open_set_hash.add(neighbor)
					neighbor.make_open()

		draw()

		if current != start:
			current.make_closed()

	return False