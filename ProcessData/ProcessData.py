#!/usr/bin/python

# remove warnings
import warnings

import inline as inline
import matplotlib as matplotlib

warnings.filterwarnings('ignore')


import pandas as pd
pd.options.display.max_columns = 100
from matplotlib import pyplot as plt
import matplotlib
matplotlib.style.use('ggplot')
import numpy as np

pd.options.display.max_rows = 100



def getCSVData(csvPath):
    data=pd.read_csv(csvPath)
    return data