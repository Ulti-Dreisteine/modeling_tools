# -*- coding: utf-8 -*-
"""
Created on 

@author: luolei


"""

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import copy


class Cell(object):
    """
    定义元胞
    """
    def __init__(self, center = np.array([0, 0]), state = 'dead'):
        """
        初始化
        :param center: 该元胞中心
        :param state: 该元胞状态
        """
        self.center = center
        self.state = state
        self.neighbors = dict()

    def get_neighbors(self, cells, pool_size):
        """
        获得该元胞四周的上下左右四个方向的邻居元胞信息
        :param cells: 细胞总体信息字典
        :param 环境尺寸
        :return:
        """
        neighbors = dict()
        for direc in [0, 1]:
            neighbors[direc] = dict()
            for delta_loc in [-1, 1]:
                neighbor_loc = copy.deepcopy(self.center)
                neighbor_loc[direc] = neighbor_loc[direc] + delta_loc
                if (neighbor_loc[direc] < 0) | (neighbor_loc[direc] > pool_size[direc] - 1):
                    neighbors[direc][delta_loc] = Cell(center = np.array([-1, -1]),
                                                       state = 'not exist')
                else:
                    neighbors[direc][delta_loc] = cells[neighbor_loc[0]][neighbor_loc[1]]
        self.neighbors = neighbors

    def show_neighbors_states(self):
        """
        显示邻居的存活状态
        :return:
        """
        print([self.neighbors[0][-1].state,
               self.neighbors[0][1].state,
               self.neighbors[1][-1].state,
               self.neighbors[1][1].state])

    def update_state(self):
        """更新该cell的状态"""
        alive_num = 0
        for direc in [0, 1]:
            for delta_loc in [-1, 1]:
                if self.neighbors[direc][delta_loc].state == 'alive':
                    alive_num += 1

        self.alive_neighbors_num = alive_num

        if (self.alive_neighbors_num >= 2):
            self.state = 'dead'
        elif (self.alive_neighbors_num >= 1) & (self.alive_neighbors_num < 2):
            self.state = 'alive'
        elif self.alive_neighbors_num < 1:
            self.state = 'dead'

class Pool(object):
    """
    元胞环境
    """
    def __init__(self, pool_size = np.array([10, 10])):
        """
        初始化pool
        :param pool_size:
        """
        self.pool_size = pool_size

        "初始化pool"
        self.cells = dict()
        for i in range(self.pool_size[0]):
            self.cells[i] = dict()
            for j in range(self.pool_size[1]):
                self.cells[i][j] = Cell(center = np.array([i, j]), state = 'dead')

    def init_alive_cells(self, init_alive_cell_locs):
        """
        初始化活cell
        :param init_cell_loc_list:
        :return:
        """
        for loc in init_alive_cell_locs:
            self.cells[loc[0]][loc[1]] = Cell(center = np.array(loc), state = 'alive')

    def update_cells_states(self):
        """
        更新pool中各cell的存活信息
        :return:
        """
        "获取所有邻居cell信息"
        new_cells = dict()
        for i in range(self.pool_size[0]):
            new_cells[i] = dict()
            for j in range(self.pool_size[1]):
                cell = Cell(center = np.array([i, j]), state = self.cells[i][j].state)
                cell.get_neighbors(self.cells, self.pool_size)
                new_cells[i][j] = cell

        "更新各cell信息"
        for i in range(self.pool_size[0]):
            for j in range(self.pool_size[1]):
                cell = new_cells[i][j]
                cell.update_state()
                new_cells[i][j] = cell
        self.cells = new_cells

    def run_simulation(self, steps = 5, show_plot = False):
        """
        进行模拟
        :param steps:
        :param show_plot:
        :return:
        """
        if show_plot == True:
            plt.figure(figsize = [6, 6])
            sns.set_style('darkgrid')

        for step in range(steps):
            "进行各cell的更新"
            self.update_cells_states()
            if show_plot == True:
                cells_locs = []
                colors_list = []
                for i in range(self.pool_size[0]):
                    for j in range(self.pool_size[1]):
                        cells_locs.append([i, j])
                        if self.cells[i][j].state == 'alive':
                            colors_list.append('w')
                        else:
                            colors_list.append('k')

                plt.clf()
                plt.scatter([p[0] for p in cells_locs],
                            [p[1] for p in cells_locs],
                            c = colors_list,
                            marker = 's',
                            s = 60,
                            edgecolors = 'k')
                plt.hold(True)
                plt.legend(['step = %s' % step])
                plt.tight_layout()
                plt.show()
                plt.pause(0.1)


if __name__ == '__main__':
    Pool = Pool(pool_size = np.array([100, 100]))
    init_alive_cell_locs = [[50, 51],
                            [51, 51],
                            [52, 51],
                            [51, 52],
                            [51, 50],
                            [50, 52],
                            [52, 50]]
    Pool.init_alive_cells(init_alive_cell_locs)

    Pool.run_simulation(steps = 500, show_plot = True)

    cell = Pool.cells[3][4]
    cell.get_neighbors(Pool.cells, Pool.pool_size)
    cell.show_neighbors_states()
    cell.update_state()