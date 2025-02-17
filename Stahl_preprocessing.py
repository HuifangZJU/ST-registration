import math
import time
import pandas as pd
import numpy as np
import scanpy as sc
import seaborn as sns
import matplotlib.pyplot as plt

import matplotlib.patches as mpatches
from matplotlib import style
import paste as pst


test = sc.read_h5ad('/media/huifang/data/registration/Stahl/Layer1_BC_ST.h5ad')
print(test)