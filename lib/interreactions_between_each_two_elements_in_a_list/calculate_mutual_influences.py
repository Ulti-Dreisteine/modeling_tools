# -*- coding: utf-8 -*-
"""
Created on Mon Oct  8 14:32:52 2018

@author: luolei

计算list中两两元素间的相互作用
"""

import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings('ignore')


class CalMutualInfluences(object):
    """
    计算list内部元素间两两相互欧式距离
    """
    def __init__(self, array):
        """
        初始化
        :param array: 待计算的array, 待计算的元素按列排列
        """
        self.array = array
        self.shape = array.shape
        self.sample_dim = self.shape[0]
        self.sample_num = self.shape[1]

    def cal_matrix_C(self):
        """
        生成中间矩阵C
        :return:
        """
        self.C = list()
        for i in range(self.sample_num):
            if i % 100 == 0:
                print('step 0: calculating matrix C, proceeding %s' % i)
            m = np.zeros([self.sample_num, 1])
            m[i] = 1
            m = np.eye(self.sample_num, self.sample_num) - m * np.ones(self.sample_num)
            self.C.append(m)

    def cal_matrix_B(self):
        """
        计算矩阵B
        :return:
        """
        self.B = list()
        for i in range(self.sample_num):
            if i % 100 == 0:
                print('step 1: calculating matrix B, proceeding %s' % i)
            self.B.append(np.dot(self.array, self.C[i]))
        del self.C

    def cal_results(self):
        self.cal_matrix_C()
        self.cal_matrix_B()
        results = list()
        for i in range(self.sample_num):
            if i % 100 == 0:
                print('step 2: calculating results, proceeding %s' % i)
            results.append(np.diag(np.dot(self.B[i].T, self.B[i])))
        results = np.array(results).reshape(self.sample_num, self.sample_num)
        del self.B
        return results


if __name__ == '__main__':
    # 生成数据
    file_name = 'data_sample.csv'
    data = pd.read_csv(file_name).head(1000)

    selected_columns = ['attr_feature_n1', 'attr_feature_n2', 'attr_feature_n3', 'attr_feature_n4']
    data = data[selected_columns]

    data = np.array(data).T

    # 测试程序
    self = CalMutualInfluences(data)
    results = self.cal_results()