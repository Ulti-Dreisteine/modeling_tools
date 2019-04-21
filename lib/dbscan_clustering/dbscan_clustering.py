# -*- coding: utf-8 -*-
"""
Created on Mon Oct  8 14:32:52 2018

@author: luolei

使用DBSCAN算法进行密度聚类
"""

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')


class DBSCAN(object):
    """
    DBSCAN算法进行密度聚类
    """
    def __init__(self, data):
        """
        初始化
        :param data: np.array格式
        :return:
        """
        self.data = data
        self.n_sample, self.dim = data.shape
        self.sample_index = np.arange(self.n_sample) # 样本索引
        self.distance_matrix = np.zeros([self.n_sample, self.n_sample]) # 距离矩阵

    @staticmethod
    def cal_distance(v_a, v_b):
        """
        计算两个向量的距离
        :param v_a:
        :param v_b:
        :return:
        """
        if len(v_a) != len(v_b):
            return Exception
        else:
            return np.sqrt(np.sum(np.square(v_a - v_b)))

    def cal_distance_matrix(self):
        """
        计算整个数据集上的距离矩阵
        :return:
        """
        for i in range(self.n_sample):
            for j in range(i + 1, self.n_sample):
                self.distance_matrix[i, j] = self.cal_distance(self.data[i], self.data[j])
                self.distance_matrix[j, i] = self.distance_matrix[i, j]

    def dbscan(self, s, minpts):  # 密度聚类
        self.cal_distance_matrix()
        center_points = []  # 存放最终的聚类结果
        k = 0  # 检验是否进行了合并过程

        for i in range(self.n_sample):
            if sum(self.distance_matrix[i] <= s) >= minpts:  # 查看距离矩阵的第i行是否满足条件
                if len(center_points) == 0:  # 如果列表长为0，则直接将生成的列表加入
                    center_points.append(list(self.sample_index[self.distance_matrix[i] <= s]))
                else:
                    for j in range(len(center_points)):  # 查找是否有重复的元素
                        if set(self.sample_index[self.distance_matrix[i] <= s]) & set(center_points[j]):
                            center_points[j].extend(list(self.sample_index[self.distance_matrix[i] <= s]))
                            k = 1  # 执行了合并操作
                    if k == 0:
                        center_points.append(list(self.sample_index[self.distance_matrix[i] <= s]))  # 没有执行合并说明这个类别单独加入
                    k = 0

        lenc = len(center_points)

        # 以下这段代码是进一步查重，center_points中所有的列表并非完全独立，还有很多重复
        # 那么为何上面代码已经查重了，这里还需查重，其实可以将上面的步骤统一放到这里，但是时空复杂的太高
        # 经过第一步查重后，center_points中的元素数目大大减少，此时进行查重更快！
        k = 0
        for i in range(lenc - 1):
            for j in range(i + 1, lenc):
                if set(center_points[i]) & set(center_points[j]):
                    center_points[j].extend(center_points[i])
                    center_points[j] = list(set(center_points[j]))
                    k = 1

            if k == 1:
                center_points[i] = []  # 合并后的列表置空
            k = 0

        center_points = [s for s in center_points if s != []]  # 删掉空列表即为最终结果
        self.center_points = center_points

        return center_points

    def show_results_plot(self):
        """
        展示聚类结果
        :return:
        """
        from matplotlib import cm
        c_n = self.center_points.__len__()  # 聚类完成后的类别数目
        color = cm.rainbow(np.linspace(0.0, 1.0, c_n))
        noise_point = np.arange(self.n_sample)  # 没有参与聚类的点即为噪声点
        for i in range(c_n):
            ct_point = list(set(self.center_points[i]))
            noise_point = set(noise_point) - set(self.center_points[i])
            print(ct_point.__len__())  # 输出每一类的点个数
            print(ct_point)  # 输出每一类的点
            print("**********")

        plt.figure(figsize = [6, 6])
        sns.set_style('darkgrid')
        for i in range(c_n):
            ct_point = list(set(self.center_points[i]))
            plt.scatter(self.data[ct_point, 0], self.data[ct_point, 1], color = color[i])  # 画出不同类别的点
        plt.grid(True)
        plt.xlabel('loc x')
        plt.ylabel('loc y')
        plt.title('DBSCAN Clustering Results')
        plt.show()


if __name__ == '__main__':
    from sklearn.datasets.samples_generator import make_blobs
    data, labels = make_blobs(n_samples = 2000,
                              centers = 7,
                              n_features = 5,
                              cluster_std = 1,
                              random_state = 0)

    # 进行DBSCAN聚类
    self = DBSCAN(data)
    center_points = self.dbscan(2, 10)
    self.show_results_plot()