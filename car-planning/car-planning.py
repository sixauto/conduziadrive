import pygame
import pygame_menu
import math
from queue import PriorityQueue

from modules.node import Node
from modules.graphic_util import *

from algorithms.astar import *
from algorithms.dijkstras import *
from algorithms.gbfs import *

WIDTH = 1920
HEIGHT = 1080
ROWS = 80  # ROW / WIDTH should be an integer (48, 60, 64, 80, 96, 120, 128, 160)
WIN = pygame.display.set_mode((WIDTH, HEIGHT))

pygame.display.set_caption("Car planning")


class Algorithms:
    DIJKSTRAS = 1
    ASTAR = 2
    GBFS = 3

    @classmethod
    def get_algorithm(cls, value: int):
        if value == cls.DIJKSTRAS:
            return dijkstrasWithVisualizer
        if value == cls.ASTAR:
            return aStarWithVisualizer
        if value == cls.GBFS:
            return GBFSWithVisualizer


class Environments:
    FACTORY = 1
    CITY = 2

    @classmethod
    def get_environment(cls, value: int):
        if value == cls.FACTORY:
            return 'factory_grid'
        if value == cls.CITY:
            return 'city_grid'


class Main:
    SELECTED_ALGORITHM = 1
    SELECTED_ENVIRONMENT = 1

    @classmethod
    def set_algorithm(cls, value, algorithm):
        cls.SELECTED_ALGORITHM = algorithm

    @classmethod
    def set_environment(cls, value, environment):
        cls.SELECTED_ENVIRONMENT = environment

    def __init__(self, WIN):
        self.__start_node = None
        self.__end_node = None
        self.__grid = make_grid(ROWS, WIDTH)
        self.__win = WIN
        if self.SELECTED_ENVIRONMENT is not None:
            self.__grid = load_grid_from_file(Environments.get_environment(self.SELECTED_ENVIRONMENT))

    def __left_button_click(self):
        pos = pygame.mouse.get_pos()
        row, col = get_clicked_pos(pos, ROWS, WIDTH)
        node = self.__grid[row][col]
        if not self.__start_node and node != self.__end_node:
            self.__start_node = node
            self.__start_node.make_start()

        elif not self.__end_node and node != self.__start_node:
            self.__end_node = node
            self.__end_node.make_end()

        elif node != self.__end_node and node != self.__start_node:
            node.make_barrier()

    def __right_button_click(self):
        pos = pygame.mouse.get_pos()
        row, col = get_clicked_pos(pos, ROWS, WIDTH)
        node = self.__grid[row][col]
        node.reset()
        if node == self.__start_node:
            self.__start_node = None
        elif node == self.__end_node:
            self.__end_node = None

    def __run_algorithm(self):
        for row in self.__grid:
            for node in row:
                node.update_neighbors(self.__grid)
        time_before = pygame.time.get_ticks() / 1000
        algorithm = Algorithms.get_algorithm(self.SELECTED_ALGORITHM)
        algorithm(lambda: draw(self.__win, self.__grid, ROWS, WIDTH), self.__grid, self.__start_node,
                  self.__end_node)
        time_after = pygame.time.get_ticks() / 1000
        show_elapsed_time(self.__win, time_before, time_after)

    def main(self):
        print("early " + str(self.SELECTED_ALGORITHM))

        run = True
        while run:
            draw(self.__win, self.__grid, ROWS, WIDTH)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    run = False

                if pygame.mouse.get_pressed()[0]:  # LEFT
                    self.__left_button_click()

                elif pygame.mouse.get_pressed()[2]:  # RIGHT
                    self.__right_button_click()

                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE and self.__start_node and self.__end_node:
                        clean_grid_except_start_end_barriers(self.__grid)
                        draw(self.__win, self.__grid, ROWS, WIDTH)

                    if (event.key == pygame.K_z or event.key == pygame.K_SPACE) and self.__start_node and self.__end_node:
                        self.__run_algorithm()

                    if event.key == pygame.K_c:
                        self.__start_node = None
                        self.__end_node = None
                        self.__grid = make_grid(ROWS, WIDTH)

                    if event.key == pygame.K_m:
                        menu(WIN)
        save_grid_to_file(self.__grid)
        pygame.quit()


def menu(win):
    pygame.init()
    pymenu = pygame_menu.Menu(HEIGHT, WIDTH, 'ConduzIA', theme=pygame_menu.themes.THEME_BLUE)
    pymenu.add_selector('Algorithm :', [('Dijkstra', 1), ('    A*   ', 2), ('   GBFS  ', 3)],
                        onchange=Main.set_algorithm)
    pymenu.add_selector('Environment :', [('Factory', 1), ('   City  ', 2)],
                        onchange=Main.set_environment)
    pymenu.add_button('Start', start)
    pymenu.add_button('Quit', pygame_menu.events.EXIT)

    pymenu.mainloop(win)


def start():
    main_session = Main(WIN)
    main_session.main()


menu(WIN)
