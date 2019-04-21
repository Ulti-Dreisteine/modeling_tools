# -*- coding: utf-8 -*-
"""
Created on Tue Jul 24 16:58:07 2018

@author: Lei Luo

Particle Swarm Optimization, PSO粒子群优化算法程序
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import random
sns.set_style('whitegrid')


def CalObjective(x = []):
    """
    定义目标函数, 向最小方向优化
    【注意】
        目标函数在此处定义
    """
#    return x[0]**2 + x[1] ** -1 + x[2] ** 3 + x[3] ** 4
#    return np.sin(x[0]) + np.cos(x[1]) ** 2 + np.sin(x[2]) ** 2 + np.cos(x[3])
    return sum([pow(x[i], 2) for i in range(len(x))])


class PSO_optimization:
    def __init__(self, obj_func = None, dim = int(), iter_num = int(), particle_num = int(), c1 = float(), \
                 c2 = float(), w = float(), dt = float()):
        self.obj_func = obj_func
        self.dim = dim
        self.iter_num = iter_num
        self.particle_num = particle_num
        self.c1 = c1
        self.c2 = c2
        self.w = w
        self.dt = dt

    def set_loc_bounds(self, loc_bounds = list()):
        self.loc_bounds = loc_bounds
    
    def set_vel_bounds(self, vel_bounds = list()):
        self.vel_bounds = vel_bounds
        
    def init_particle_locs(self):
        """
        初始化粒子位置
        【输入】
            :dim, 所研究优化问题的的变量数
            :loc_bounds, 各变量的上下界
                loc_bounds = list([[lb_0, ub_0], [lb_1, ub_1], ..., [lb_dim, ub_dim]])
        【输出】
            :particle_locs, dict类型, 键为粒子编号
        """
        self.particle_locs = dict()
        for i in range(self.particle_num):
            self.particle_locs[i] = list()
            for j in range(self.dim):
                self.particle_locs[i].append(random.random() * (max(self.loc_bounds[j]) - min(self.loc_bounds[j])) + min(self.loc_bounds[j]))
    
    def init_particle_vels(self):
        """
        初始化粒子速度
        【输入】
            :dim, 所研究优化问题的的变量数
            :vel_bounds, 各变量的上下界
                vel_bounds = list([[lb_0, ub_0], [lb_1, ub_1], ..., [lb_dim, ub_dim]])
        【输出】
            :particle_vels, dict类型, 键为粒子编号
        """
        self.particle_vels = dict()
        for i in range(self.particle_num):
            self.particle_vels[i] = list()
            for j in range(self.dim):
                self.particle_vels[i].append(random.random() * (max(self.vel_bounds[j]) - min(self.vel_bounds[j])) + min(self.vel_bounds[j]))

    def pso_optimization(self, show_results = False):
        """
        粒子群优化算法, 计算给定范围内的全局最小值
        【输入】
            :dim, 待优化变量维度, int类型
            :iter_num, 迭代次数, int类型
            :particle_num, 粒子数目, int类型
            :c1, 局部学习因子, float类型
            :c2, 全局学习因子, float类型
            :w, 惯性权重, float类型
            :dt, 时间步长, float类型
            :loc_bounds, 粒子各变量各坐标取值边界, list类型
            :vel_bounds, 粒子速度边界, list类型
            :show_results, 是否显示粒子运动图和目标函数下降过程, bool类型
        【调用】
            :InitializeParticleLocs, 初始化各粒子位置
            :InitializeParticleVels, 初始化各粒子速度
            :CalObjective, 计算目标函数值
        【返回】
            global_best_locs, 全局最优解, float类型
            global_best_value, 全局最优(小)值, float类型
            global_opt_obj, 目标函数变化记录, list类型
        """
        '初始化各粒子位置和速度'
        particle_locs = self.particle_locs
        particle_vels = self.particle_vels
        dim = self.dim
        particle_num = self.particle_num
        iter_num =self.iter_num
        w = self.w
        c1 = self.c1
        c2 = self.c2
        dt = self.dt
        
        
        '将各粒子位置和速度以pd.DataFrame形式保存'
        particle_locs = pd.DataFrame(particle_locs).T
        particle_locs.columns = list(['loc_' + str(i) for i in range(dim)])
        particle_locs['particle_num'] = particle_locs.index
        particle_vels = pd.DataFrame(particle_vels).T
        particle_vels.columns = list(['vel_' + str(i) for i in range(dim)])
        particle_vels['particle_num'] = particle_vels.index
        
        loc_column_list = list()
        vel_column_list = list()
        new_vel_column_list = list()
        new_loc_column_list = list()
        for j in range(dim):
            loc_column_list.append('loc_' + str(j))
            vel_column_list.append('vel_' + str(j))
            new_vel_column_list.append('new_vel_' + str(j))
            new_loc_column_list.append('new_loc_' + str(j))
        
        particle_vars = pd.merge(particle_locs, particle_vels, how = 'left', left_on = 'particle_num', right_on = 'particle_num')
        particle_vars['objective'] = particle_vars[loc_column_list].apply(self.obj_func, axis = 1)
    
        global_opt_obj = list()
        obj_results_each_iter = list()
        '其他可能需要输出的结果'
        if show_results == True:
            particle_locus = list()
           
        iter = 0
        while iter < iter_num:   
            '寻找本轮迭代中的最优值和位置以及截至目前的最优值和位置'
            if iter == 0:
                iter_best_value = particle_vars['objective'].min()
                iter_best_locs = particle_vars[particle_vars['objective'] == iter_best_value][loc_column_list]
                global_best_value = particle_vars['objective'].min()
                global_best_locs = particle_vars[particle_vars['objective'] == global_best_value][loc_column_list]
            else:
                iter_best_value = particle_vars['objective'].min()
                iter_best_locs = particle_vars[particle_vars['objective'] == iter_best_value][loc_column_list]
                '判断是否执行global值的更新'
                if iter_best_value < global_best_value:
                    global_best_value = iter_best_value
                    global_best_results = particle_vars[particle_vars['objective'] == global_best_value]
                    global_best_locs = pd.DataFrame(global_best_results[loc_column_list].iloc[0]).T
            
            '1. 执行速度更新'
            def UpdateVels(x):
                """
                各维度下的速度值更新
                """
                loc_label = vel_dim_num
                vel_label = vel_dim_num + dim
                vel_new = w * x[vel_label] +\
                c1 * random.random() * (iter_best_locs.iloc[0]['loc_' + str(vel_dim_num)] - x[loc_label]) +\
                c2 * random.random() * (global_best_locs.iloc[0]['loc_' + str(vel_dim_num)] - x[loc_label])
                return vel_new
            
            loc_vel_column_list = loc_column_list + vel_column_list
            for i in range(dim):
                vel_dim_num = i
                particle_vars['new_vel_' + str(vel_dim_num)] = particle_vars[loc_vel_column_list].apply(UpdateVels, axis = 1)
            
            '2. 执行位置更新'
            def UpdateLocs(x):
                """
                各维度下的位置坐标更新
                """
                return x[loc_dim_num] + dt * x[loc_dim_num + dim]
            
            for i in range(dim):
                loc_dim_num = i
                loc_new_vel_column_list = loc_column_list + new_vel_column_list
                particle_vars['new_loc_' + str(loc_dim_num)] = particle_vars[loc_new_vel_column_list].apply(UpdateLocs, axis = 1)
            
            '3. 计算目标函数值'            
            particle_vars['new_objective'] = particle_vars[new_loc_column_list].apply(self.obj_func, axis = 1)
            
            '4. 替换原有的位置速度和目标函数值'
            def Substitute(x):
                """
                位置loc, 速度vel和目标函数值objective的替换
                """
                return x[1]
            
            columns_to_substitute = loc_column_list + vel_column_list + list(['objective'])
            for column in columns_to_substitute:
                particle_vars[column] = particle_vars[[column, 'new_' + column]].apply(Substitute, axis = 1)
            
            global_opt_obj.append(global_best_value)
            obj_results_each_iter.append(list(particle_vars['objective']))
            if show_results == True:
                particle_locus.append(particle_vars[loc_column_list].to_dict())
            print('这是第%s次迭代, 目标函数值为%.4f' % (str(iter), global_best_value))
            
            iter += 1
        
        if show_results == True:
            plt.figure('objective')
            plt.plot(global_opt_obj)
            plt.xlabel('iteration steps')
            plt.ylabel('objective value')
            
            if particle_num > 20:
                print('颗粒数目太多, 无法对particle_locus和obj_results_at_each_iter作图')
            elif particle_num <= 1:
                print('颗粒数目太少, 无法对大于一维问题的particle_locus和obj_results_at_each_iter作图')
            else:
                particle_name_list = list()
                for i in range(particle_num):
                    particle_name_list.append('particle_' + str(i))
                
                plt.figure('show results', figsize = list([8, 4]))
                plt.subplot(1, 2, 1)
                plt.title('particle locus')
                plt.scatter(global_best_locs.iloc[0]['loc_0'], global_best_locs.iloc[0]['loc_1'], c = 'k', s = 30)
                plt.hold(True)
                for i in range(particle_num):
                    plt.plot([particle_locus[j]['loc_0'][i] for j in range(iter_num)], [particle_locus[j]['loc_1'][i] for j in range(iter_num)], linewidth = 1)
                    plt.hold(True)
                plt.legend(particle_name_list + ['opt_position'], fontsize = 6)
                
                plt.subplot(1, 2, 2)
                plt.title('objective at each iteration step')
                for i in range(particle_num):
                    plt.plot(range(iter_num), [obj_results_each_iter[j][i] for j in range(iter_num)], linewidth = 1)
                    plt.xlabel('iteration steps')
                    plt.ylabel('objective value')
                    plt.hold(True)
                plt.legend(particle_name_list, fontsize = 6)
        
        return global_best_locs, global_best_value, global_opt_obj, obj_results_each_iter
       

if __name__ == '__main__':
    pso_opt = PSO_optimization()
    
    '参数赋值'
    pso_opt.dim = 4
    pso_opt.iter_num = 500
    pso_opt.particle_num = 10
    pso_opt.c1 = 0.5
    pso_opt.c2 = 0.5
    pso_opt.w = 0.8
    pso_opt.dt = 0.2
    pso_opt.obj_func = CalObjective
    
    '指定位置和速度的上下界'
    loc_bounds = list()
    vel_bounds = list()
    for i in range(pso_opt.dim):
        loc_bounds.append([-10, 10])
        vel_bounds.append([-0.1, 0.1])
    
    pso_opt.set_loc_bounds(loc_bounds)
    pso_opt.set_vel_bounds(vel_bounds)
    
    '初始化粒子的位置和速度'
    pso_opt.init_particle_locs()
    pso_opt.init_particle_vels()
    
    '进行PSO优化'
    global_best_locs, global_best_value, global_opt_obj, obj_results_each_iter = pso_opt.pso_optimization(show_results = True)
        
        