import pygame
from modules.color import Color

class Node:

	def __init__(self, row, col, width, total_rows):
		self.row = row
		self.col = col
		self.x = row * width
		self.y = col * width
		self.color = Color.WHITE
		self.neighbors = []
		self.width = width
		self.total_rows = total_rows

	def get_pos(self):
		return self.row, self.col

	def is_closed(self):
		return self.color == Color.BLUE

	def is_open(self):
		return self.color == Color.LIGHT_BLUE

	def is_barrier(self):
		return self.color == Color.DARK_BLUE

	def is_start(self):
		return self.color == Color.ORANGE

	def is_path(self):
		return self.color == Color.LIGHT_YELLOW

	def is_end(self):
		return self.color == Color.GREEN

	def reset(self):
		self.color = Color.WHITE

	def make_start(self):
		self.color = Color.ORANGE

	def make_closed(self):
		self.color = Color.BLUE

	def make_open(self):
		self.color = Color.LIGHT_BLUE

	def make_barrier(self):
		self.color = Color.DARK_BLUE

	def make_end(self):
		self.color = Color.GREEN

	def make_path(self):
		self.color = Color.LIGHT_YELLOW

	def draw(self, win):
		pygame.draw.rect(win, self.color, (self.x, self.y, self.width, self.width))
  
	def update_neighbors(self, grid):
		self.neighbors = []
		if self.row < self.total_rows - 1 and not grid[self.row + 1][self.col].is_barrier(): # DOWN
			self.neighbors.append(grid[self.row + 1][self.col])

		if self.row > 0 and not grid[self.row - 1][self.col].is_barrier(): # UP
			self.neighbors.append(grid[self.row - 1][self.col])

		if self.col < self.total_rows - 1 and not grid[self.row][self.col + 1].is_barrier(): # RIGHT
			self.neighbors.append(grid[self.row][self.col + 1])

		if self.col > 0 and not grid[self.row][self.col - 1].is_barrier(): # LEFT
			self.neighbors.append(grid[self.row][self.col - 1])
