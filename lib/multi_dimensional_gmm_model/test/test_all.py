# -*- coding: utf-8 -*-
"""
Created on Mon Oct  8 14:32:52 2018

@author: luolei


"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from nose.tools import *

from main.optimize_2d_cov import *


class Test(object):
	def setup(self):
		self.vars = [1, 3]
		self.corner = 1
		self.cov = cov_matrix(self.vars, self.corner)
	
	def teardown(self):
		pass
	
	def test_ort_matrix(self):
		U = ort_matrix(self.cov)
		ort_cov = np.dot(np.dot(U.T, self.cov), U)
		assert_less(ort_cov[0][1], 1e-6)
		assert_less(ort_cov[1][0], 1e-6)
	
if __name__ == '__main__':
	test = Test()
	test.setup()
	test.teardown()
	test.test_ort_matrix()
	
	


