# -*- coding: utf-8 -*-
"""
Created on Fri Jul 27 09:41:34 2018

@author: Lei Luo

遗传算法
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import seaborn as sns

sns.set_style('whitegrid')


def normalize(x):
    """
    归一化x
    :param x: list
    :return:
    """
    return list([p / sum(x) for p in x])


class GA_Optimization:
    def __init__(self, obj_function = None, dim = int(), iter_num = int(), pop_size = int(),
                 chrom_length = int(), pc = float(0.6), pm = float(0.01), adjust_factor = float(5.0),\
                 show_progress = False):
        """
        初始化
        【输入】
            :obj_function, 目标函数
            :dim, 自变量维数, int类型
            :iter_num, 优化迭代的次数, int类型
            :pop_size, 种群数量
            :chrom_length, 染色体长度
            :pc, 交叉概率
            :pm, 变异概率
            :adjust_factor, 归一化调整因子
            :show_progress, 是否显示进化过程中目标函数变化
        """
        self.obj_function = obj_function
        self.dim = dim
        self.iter_num = iter_num
        self.pop_size = pop_size
        self.chrom_length = chrom_length
        self.pc = pc
        self.pm =pm
        self.adjust_factor = adjust_factor
        self.show_progress = show_progress

    def init_pop(self):
        """
        根据染色体长度和种群数量产生染色体种群
        【返回】
            :pop, 染色体种群, 同一染色体上各基因值之和为1, pd.DataFrame结构, 行为基因位置, index为个体编号
        """
        pop = list([list([])])  
        for i in range(self.pop_size):  
            temp = []  
            for j in range(self.chrom_length):  
                temp.append(random.random())
                if self.chrom_length >= 2:
                    temp = [p / sum(temp) for p in temp]
            pop.append(temp)   
        self.pop = pd.DataFrame(pop[1:])
        
    def cal_obj_value(self):
        """
        计算种群中各个体的目标函数值
        """
        gene_column_list = list(range(self.chrom_length))
        self.pop['obj_value'] = self.pop[gene_column_list].apply(self.obj_function, axis = 1)
        
    def zero_one_normalize_obj_value(self):
        def zero_one_normalize(x):
            if self.pop['obj_value'].min() == self.pop['obj_value'].max():
                return random.random()
            else:
                return (x - self.pop['obj_value'].min()) / (self.pop['obj_value'].max() - self.pop['obj_value'].min())
        self.pop['zero_one_normed_obj_value'] = self.pop['obj_value'].apply(zero_one_normalize)
        
    def cal_fit_value(self):
        """
        计算种群中各染色体个体的适应度函数值
        """
        def neg_adjust_value(x):
            """
            计算适应度函数
            """
            return (np.exp(-self.adjust_factor * x) - np.exp(-self.adjust_factor)) / (1 - np.exp(-self.adjust_factor))
        self.pop['fit_value'] = self.pop['zero_one_normed_obj_value'].apply(neg_adjust_value)
    
    def zero_one_normalize_fit_value(self):
        def zero_one_normalize(x):
            if self.pop['fit_value'].min() == self.pop['fit_value'].max():
                return random.random()
            else:
                return (x - self.pop['fit_value'].min()) / (self.pop['fit_value'].max() - self.pop['fit_value'].min())
        self.pop['zero_one_normed_fit_value'] = self.pop['fit_value'].apply(zero_one_normalize)
    
    def adjust_fit_value(self):
        def pos_adjust_value(x):
            """
            计算适应度函数
            """
            return (np.exp(self.adjust_factor * x) - 1) / (np.exp(self.adjust_factor) - 1)
        self.pop['adjusted_fit_value'] = self.pop['zero_one_normed_fit_value'].apply(pos_adjust_value)
    
    def normalize_adjusted_fit_value(self):
        sum_adjusted_fit_value = self.pop['adjusted_fit_value'].sum()
        def normalize_adjust(x):
            """
            对适应度函数进行归一化, 便于之后的概率计算
            """
            return x / sum_adjusted_fit_value
        self.pop['normed_adjusted_fit_value'] = self.pop['adjusted_fit_value'].apply(normalize_adjust)
    
    def find_best_individual(self):
        """
        寻找当前迭代过程的种群pop中的最大适应度以及对应的最适个体染色体基因组
        【返回】
            :best_fit_value, 最佳适应度值, float类型
            :best_fit_individual, 最适应个体染色体基因组
        """
        self.best_obj_value = self.pop['obj_value'].min()
        self.best_fit_value = self.pop['fit_value'].max()
        self.best_fit_individual = pd.DataFrame(pd.DataFrame(self.pop[self.pop['fit_value'] == self.best_fit_value]).iloc[0][list(range(self.chrom_length))]).T   
        
    def select(self):
        """
        挑选下一代种群个体染色体基因组
        """
        self.zero_one_normalize_fit_value()
        self.adjust_fit_value()
        self.normalize_adjusted_fit_value()
        
        '使用轮盘算法选择个体'
        '累积概率表'
        accumulate_prob_list = list([sum(self.pop.iloc[0 : i]['normed_adjusted_fit_value']) for i in range(self.pop_size + 1)])
        
        self.new_pop = dict()
        for i in range(self.pop_size):
            random_num = random.random()
            if random_num == 0.0:
                self.new_pop[i] = list(self.pop.iloc[0])
            else:
                for j in range(len(accumulate_prob_list)):
                    if (j <= self.pop_size - 1):
                        if (random_num > accumulate_prob_list[j]) & (random_num <= accumulate_prob_list[j + 1]): 
                            self.new_pop[i] = list(self.pop.iloc[j])
                            break
                    else:
                        self.new_pop[i] = list(self.pop.iloc[self.pop_size - 1])
                        break
        self.new_pop = pd.DataFrame(self.new_pop).T
        self.new_pop.columns = list(range(self.chrom_length)) + ['obj_value', \
                                   'zero_one_normed_obj_value', 'fit_value', \
                                   'zero_one_normed_fit_value', 'adjusted_fit_value', \
                                   'normed_adjusted_fit_value']
        self.pop = self.new_pop
        
    def cross_over(self):
        """
        执行交配操作
        """       
        for i in range(self.pop_size - 1):  
            if (random.random() < self.pc): 
                '定义交叉点'
                cpoint = random.randint(0, self.chrom_length)
                self.pop.iloc[i][cpoint], self.pop.iloc[i + 1][cpoint] = self.pop.iloc[i + 1][cpoint], self.pop.iloc[i][cpoint]
                
                '注意同一染色体中基因值之和归一化' 
                for j in range(i, i + 2):
                    normalized_gene_value_list = list(pd.DataFrame(self.pop.iloc[j]).T[list(range(self.chrom_length))].apply(normalize, axis = 1))[0]
                    for k in list(range(self.chrom_length)):
                        self.pop.iloc[j][k] = normalized_gene_value_list[k]
        self.cal_obj_value()
        self.zero_one_normalize_obj_value()
        self.cal_fit_value()
        self.zero_one_normalize_fit_value()
        self.adjust_fit_value()
        self.normalize_adjusted_fit_value()
            
    def mutation(self):
        """
        执行基因突变操作
        """
        for i in range(self.pop_size):
            if (random.random() < self.pm):
                mpoint = random.randint(0, self.chrom_length - 1)
                self.pop.iloc[i][mpoint] = random.random()
                '注意同一染色体中基因值之和归一化' 
                normalized_gene_value_list = list(pd.DataFrame(self.pop.iloc[i]).T[list(range(self.chrom_length))].apply(normalize, axis = 1))[0]
                for k in list(range(self.chrom_length)):
                    self.pop.iloc[i][k] = normalized_gene_value_list[k]
        self.cal_obj_value()
        self.zero_one_normalize_obj_value()
        self.cal_fit_value()
        self.zero_one_normalize_fit_value()
        self.adjust_fit_value()
        self.normalize_adjusted_fit_value()
    
    def evolution(self): 
        '产生初始种群'
        self.init_pop()
        
        '计算种群各个体适应度'
        self.cal_obj_value()
        self.zero_one_normalize_obj_value()
        self.cal_fit_value()
        
        best_fit_value = list()
        iteration = 0
        if self.show_progress == True:
            plt.figure('object value')
        while iteration < self.iter_num:
            '执行选择操作'
            self.select()
            
            '执行交配操作'
            self.cross_over()
            
            '执行突变操作'
            self.mutation()
            
            '计算种群各个体适应度'
            self.cal_fit_value()
            
            '找出当前最优适应度和对应基因组'
            self.find_best_individual()
            best_fit_value.append(self.best_fit_value)
            
            print('第%d步迭代, 目标函数为%.6f, 最适个体为%s, ...]' % (iteration, self.best_obj_value, str(list(self.best_fit_individual.iloc[0][0 : 2]))[:-1]))
            
            if self.show_progress == True:
                plt.scatter(iteration, self.best_obj_value, c = 'b', s = 2)
                plt.xlabel('iteration steps')
                plt.ylabel('current best objective value')
                plt.show()
                plt.pause(0.01)
                
            iteration += 1


if __name__ == '__main__':
    '定义目标函数'
    def ObjectiveFunction(x = []):
        """
        定义目标函数, 本程序中目标函数向最小化优化
        """
        m = 1
        for i in range(len(x)):
            m *= np.cos(x[i] / pow(i + 1, 0.5))
        return sum([pow(i, 2) / 4000 for i in x]) - m + 1
    
    x = np.arange(-10, 10, 0.05)
    y = np.arange(-10, 10, 0.05)
    results = np.zeros([len(x), len(y)])
    for i in range(len(x)):
        for j in range(len(y)):
            results[i][j] = ObjectiveFunction(list([x[i], y[j]]))
    plt.figure('Griewank')
    sns.heatmap(results, cmap = 'Blues')
    plt.pause(0.1)

    '参数赋值'
    ga_opt = GA_Optimization()
    ga_opt.pop_size = 200
    ga_opt.chrom_length = 8
    ga_opt.iter_num = 500
    ga_opt.pc = 0.4
    ga_opt.pm = 0.3
    ga_opt.adjust_factor = 5
    ga_opt.obj_function = ObjectiveFunction
    ga_opt.show_progress = True
    
    '执行进化优化'
    ga_opt.evolution()
    
    '输出种群'
    pop = ga_opt.pop
    