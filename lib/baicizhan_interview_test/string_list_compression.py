# -*- coding: utf-8 -*-
"""
Created on Mon Oct  8 14:32:52 2018

@author: luolei

百词斩笔试试题: 字符串压缩
"""

import pandas as pd
import numpy as np
import networkx as nx
import seaborn as sns
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')

# DEFINE CLASSES AND FUNCTIONS HERE ————————————————————————————————————————————————————————————————————————————————————
class StringListCompression(object):
    """
    执行字符串压缩操作, 给定由普通英文字母组成的非空字符串s1，要求将连续出现的字符压缩成字符和该字符连续出现的次数，并返回新的字
    符串s2;
    s1字符串的长度不超过100;
    """
    def __init__(self, string_list):
        """
        初始化
        :param string_list:
        """
        self.string_list = string_list
        self._check_input_format()

    def _check_input_format(self):
        """
        检查输入字符串的格式
        :return:
        """
        "长度检查"
        if len(self.string_list) > 100:
            raise ValueError("the length of the string list is too long")

        "字符类型检查"
        if self.string_list.encode('utf-8').isalpha() == False:
            raise ValueError("not all chars are English characters")

    def compress_string_list(self):
        """
        按照以上规则执行字符串压缩
        :return: compressed_str_list
        """
        "依次统计各字符出现的次数"
        compressed_str_list = ''
        for i in range(len(self.string_list)):
            if i == 0:
                count = 1
                compressed_str_list = compressed_str_list + self.string_list[i]
            else:
                if self.string_list[i] == self.string_list[i - 1]:
                    count += 1
                else:
                    compressed_str_list = compressed_str_list + str(count) * (1 - int(count == 1))
                    compressed_str_list = compressed_str_list + self.string_list[i]
                    count = 1
                if i == len(self.string_list) - 1:
                    compressed_str_list = compressed_str_list + str(count) * (1 - int(count == 1))
        return compressed_str_list


# MAIN PROGRAM —————————————————————————————————————————————————————————————————————————————————————————————————————————
if __name__ == '__main__':
    string_list = 'aab'
    self = StringListCompression(string_list)
    compressed_str_list = self.compress_string_list()