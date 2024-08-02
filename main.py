import os
import tarfile
from six.moves import urllib
import pandas as pd
import numpy as np
from zlib import crc32
from pandas.plotting import scatter_matrix
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
# downloadRoot = "https://raw.githubusercontent.com/ageron/handson-ml2/master/"
housingPath = os.path.join("data", "Housing")
# housingURL = downloadRoot + "datasets/housing/housing.tgz"

# def fetch_housing_data(housing_url=housingURL, housing_path=housingPath):
#  if not os.path.isdir(housing_path):
#    os.makedirs(housing_path)
#  tgz_path = os.path.join(housing_path, "housing.tgz")
#  urllib.request.urlretrieve(housing_url, tgz_path)
 # housing_tgz = tarfile.open(tgz_path)
 # housing_tgz.extractall(path=housing_path)
 # housing_tgz.close()

def loadHouseData(housing_path=housingPath):
 csv_path = os.path.join(housing_path, "housing.csv")
 return pd.read_csv(csv_path)

# fetch_housing_data()
# loadHouseData()
housing = loadHouseData()
# print(housing.head())
# print(housing.info())
housing.hist(bins=50, figsize=(20, 15))
plt.show()