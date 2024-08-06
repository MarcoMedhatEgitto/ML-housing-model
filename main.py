# importing libraries
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
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt
roomsIx, bedroomsIx, populaionIx, householdsIx = 3, 4, 5, 6
class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
 def __init__(self, addBedroomsPerRoom = True):
  self.addBedroomsPerRoom = addBedroomsPerRoom
 def fit(self, X, y = None):
  return self
 def transform(self, X, y=None):
  roomsPerhousehold = X[:, roomsIx]/X[:, householdsIx]
  populationPerHousehold = X[:, populaionIx]/X[:, householdsIx]
  if self.addBedroomsPerRoom:
   bedroomsPerRoom = X[:, bedroomsIx]/X[:, roomsIx]
   return np.c_[X, roomsPerhousehold, populationPerHousehold, bedroomsPerRoom]
  else:
   return np.c_[X, roomsPerhousehold, populationPerHousehold]

# declaring some variables to help in housing data download
# downloadRoot = "https://raw.githubusercontent.com/ageron/handson-ml2/master/"
housingPath = os.path.join("data", "Housing")
# housingURL = downloadRoot + "datasets/housing/housing.tgz"


# downloading housing data
# def fetch_housing_data(housing_url=housingURL, housing_path=housingPath):
#  if not os.path.isdir(housing_path):
#    os.makedirs(housing_path)
#  tgz_path = os.path.join(housing_path, "housing.tgz")
#  urllib.request.urlretrieve(housing_url, tgz_path)
 # housing_tgz = tarfile.open(tgz_path)
 # housing_tgz.extractall(path=housing_path)
 # housing_tgz.close()

# fetch_housing_data()

# loading Housing data into the model
def loadHouseData(housing_path=housingPath):
 csv_path = os.path.join(housing_path, "housing.csv")
 return pd.read_csv(csv_path)


# visualising the data
housing = loadHouseData()
# print(housing.head())
# print(housing.info())
# housing.hist(bins=50, figsize=(20, 15))
# plt.show()

#splitting into train test sets
# def splitTrainTest(data, testRatio):
#  shuffledIndices = np.random.permutation(data.index)
#  testSetSize = int(len(shuffledIndices) * testRatio)
#  testIndices = shuffledIndices[:testSetSize]
#  trainIndices = shuffledIndices[testSetSize:]
#  return data.iloc[trainIndices], data.iloc[testIndices]

# trainSet, testSet = splitTtrainTest(housing, 0.2)

# print(len(trainSet))
# print(len(testSet))
# discovered an issue, every time it wil load the datat it will have different 20% in the testing set,
# allowing the model to vissualize all the data.

# fixing issue in train/test split by getting the hash of the id and making it as test set if beneath 20%
# def testSetCheck(identifier, testRatio):
#  return crc32(np.int64(identifier)) & 0xffffffff < testRatio * 2 ** 32
# def splitTrainTestById(data, testRatio, idColumn):
#  ids=data[idColumn]
#  inTestSet = ids.apply(lambda id: testSetCheck(id, testRatio))
#  return data.loc[~inTestSet], data.loc[inTestSet]

#as if our dataset doesn't have id

# housingId = housing.reset_index()

# getting the train/test data

# trainSet, testSet = splitTrainTestById(housingId, 0.2, "index")
# print(len(trainSet))
# print(len(testSet))

#doing the split by scikit depending on income_category
trainSet, testSet = train_test_split(housing, test_size=0.2, random_state=42)
housing["income_cat"] = pd.cut(housing["median_income"],
 bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
 labels=[1, 2, 3, 4, 5])
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(housing, housing["income_cat"]):
 stratTrainSet = housing.loc[train_index]
 stratTestSet = housing.loc[test_index]

# print(stratTestSet["income_cat"].value_counts() / len(stratTestSet))

for set_ in (stratTrainSet, stratTestSet):
 set_.drop("income_cat", axis=1, inplace=True)



# getting the income category
# housing["income_cat"] = pd.cut(housing["median_income"],
 # bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
 # labels=[1, 2, 3, 4, 5])
# housing["income_cat"].hist()

#visualising the data to understand it

housing = stratTrainSet.copy()
# housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.1)
# plt.show()
# housing.plot(
#              kind="scatter",
#              x="longitude",
#              y="latitude",
#              alpha=0.4,
#              s=housing["population"]/100,
#              label="Population",
#              figsize=(10, 7),
#              c="median_house_value",
#              cmap=plt.get_cmap("jet"),
#              colorbar=True,
#              )
# plt.legend()

# discovering the correlations
# corr_matrix = housing.corr()
# corrMatrix(["median_house_value"].sort_values(ascending=False)

#preparing data to feed to the algorithm

housing = stratTrainSet.drop("median_house_value", axis=1)
housingLabels = stratTrainSet["median_house_value"].copy()
# cleaning data as the bedrooms are missing some districts
# housing.drop("total_bedrooms", axis=1)
# housing.dropna(subset=["total_bedrooms"])
# median = housing["total_bedrooms"].median()
# housing["total_bedrooms"].fillna(median, inplace=True)
# let's take care of it through scikit
# get the median to add to the missing districts in the attribute
imputer = SimpleImputer(strategy="median")
housingNm=housing.drop("ocean_proximity", axis=1)
imputer.fit(housingNm)
# print(imputer.statistics_)
# print(housingNm.median().values)
# x = imputer.transform(housingNm)
# print(x)
housingCat = housing[["ocean_proximity"]]
# print(housingCat.head(10))
# let's change the ocean category in numbers as it has string options
ordinalEncoder = OrdinalEncoder()
housingCatEncoded = ordinalEncoder.fit_transform(housingCat)
# print(housingCatEncoded[:10])
# print(ordinalEncoder.categories_)
catEncoder = OneHotEncoder()
housingCatEncoded = catEncoder.fit_transform(housingCat)
attributeAdder = CombinedAttributesAdder(addBedroomsPerRoom = False)
housingExtraAttribs = attributeAdder.transform(housing.values)
numPipeline = Pipeline([
 ('imputer', SimpleImputer(strategy="median")),
 ('attribs_adder', CombinedAttributesAdder()),
 ('std_scaler', StandardScaler()),
 ])
housingNmTr = numPipeline.fit_transform(housingNm)
numAttribs = list(housingNm)
catAttribs = ["ocean_proximity"]
fullPipline = ColumnTransformer([
                                  ("num", numPipeline, numAttribs),
                                  ("cat", OneHotEncoder(), catAttribs)
                                ])
housingPrepared = fullPipline.fit_transform(housing)

# after working on the data let's train the model using linear regression

# linReg = LinearRegression()
# linReg.fit(housingPrepared, housingLabels)

# let's try it by the trainig set

# someData = housing.iloc[:5]
# someLabels = housingLabels[:5]
# someDataPrepared = fullPipline.transform(someData)
# print("Prediction: ", linReg.predict(someDataPrepared))
# print("Labels: ", list(someLabels))

#let's measure the cost

# housingPrediction = linReg.predict(housingPrepared)
# linMSE = mean_squared_error(housingLabels, housingPrediction)
# linRMSE = np.sqrt(linMSE)
# print(linRMSE)

# let's try decision tree to see if we have better results

treeReg = DecisionTreeRegressor()
treeReg.fit(housingPrepared, housingLabels)
housingPredictions = treeReg.predict(housingPrepared)
treeMSE = mean_squared_error(housingLabels, housingPredictions)
treeRMSE = np.sqrt(treeMSE)
print(treeRMSE)
# It overfitts the data


