# functions for preprocessing data ...
# imports
import pandas as pd
import numpy as np
from skrub import TableReport
import scikit_na as na
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
