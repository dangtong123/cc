import ee
import geemap
import sys
import os
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Add local module paths
module_path = r'E:\s1\code'
if module_path not in sys.path:
    sys.path.append(module_path)

import wrapper as wp
import border_noise_correction as bnc
import speckle_filter as sf
import terrain_flattening as trf
import helper

# Initialize Earth Engine
ee.Initialize()

# Set up proxy if needed
geemap.set_proxy(port='7890')

# Create map centered on a specific location
Map = geemap.Map(center=[30.9756, 112.2707], zoom=7)
Map.add_basemap('ROADMAP')

# Asset paths
exportPath = 'projects/ee-tongdang981/assets/youhua/'
s2Composite = ee.Image(exportPath + 'sentinel1_2_landsat8_fusion')
demBands = ee.Image(exportPath + 'dem_bands')
gediMosaic = ee.Image(exportPath + 'gedi_mosaic')

# Center map on composite image geometry
geometry = s2Composite.geometry()
Map.centerObject(geometry)
Map

# Grid size and projection
gridScale = 100
gridProjection = ee.Projection('EPSG:3857').atScale(gridScale)

# Stack images and resample
stacked = s2Composite.addBands(demBands).addBands(gediMosaic)
stacked = stacked.resample('bilinear')

# Aggregate pixels and compute mean
stackedResampled = stacked.reduceResolution(
    reducer=ee.Reducer.mean(),
    maxPixels=1024
).reproject(
    crs=gridProjection
)

# Update mask to remove transparency
stackedResampled = stackedResampled.updateMask(stackedResampled.mask().gt(0))

# Define predictors and predicted bands
predictors = s2Composite.bandNames().cat(demBands.bandNames())
predicted = gediMosaic.bandNames().get(0)

print('Predictors:', predictors.getInfo())
print('Predicted:', predicted.getInfo())

predictorImage = stackedResampled.select(predictors)
predictedImage = stackedResampled.select([predicted])

print(f"Selected predictor bands: {predictorImage.bandNames().getInfo()}")
print(f"Selected predicted band: {predictedImage.bandNames().getInfo()}")

# Create classification mask
classMask = predictedImage.mask().toInt().rename('class')
print(f"Class Mask: {classMask.getInfo()}")

# Load landcover data from GEE assets
landcover = ee.Image('projects/ee-tongdang981/assets/hubeitdly')
landcover = landcover.reproject(crs=gridProjection, scale=gridScale)

# Define landcover types and sample sizes
landcoverValues = {
    'Shrubland': 3,  # Shrub
    'Forest': 2,     # Forest
    'Grassland': 4   # Grassland
}

samples = []
numSamplesPerClass = 400

for landcoverType, landcoverValue in landcoverValues.items():
    lcMask = landcover.eq(landcoverValue)
    lcSample = stackedResampled.addBands(classMask).updateMask(lcMask)
    
    print(f"Landcover Type: {landcoverType}")
    try:
        lcSampleBandNames = lcSample.bandNames().getInfo()
        print(f"LC Sample Band Names: {lcSampleBandNames}")
    except Exception as e:
        print(f"Error getting band names for {landcoverType}: {e}")
    
    try:
        lcSample = lcSample.stratifiedSample(
            numPoints=numSamplesPerClass,
            classBand='class',
            region=geometry,
            scale=gridScale,
            classValues=[0, 1],
            classPoints=[0, numSamplesPerClass],
            dropNulls=True,
            tileScale=16
        )
        samples.append(lcSample)
    except ee.EEException as e:
        print(f"Error during stratified sampling for {landcoverType}: {e}")

# Combine all samples
try:
    training = ee.FeatureCollection(samples).flatten()
    print('Number of Features Extracted', training.size().getInfo())
    print('Sample Training Feature', training.first().getInfo())
except ee.EEException as e:
    print(f"Error combining samples: {e}")

# Convert EE FeatureCollection to Pandas DataFrame
def ee_to_pandas(fc):
    try:
        features = fc.getInfo()['features']
        dict_list = [f['properties'] for f in features]
        return pd.DataFrame(dict_list)
    except ee.EEException as e:
        print(f"Error converting FeatureCollection to DataFrame: {e}")
        return pd.DataFrame()

training_df = ee_to_pandas(training)

# Extract features and labels
try:
    X = training_df[predictors.getInfo()]
    y = training_df[predicted.getInfo()]

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.15, random_state=42)
except KeyError as e:
    print(f"Error extracting features and labels: {e}")
except Exception as e:
    print(f"Unexpected error: {e}")


# 打印样本数量
print(f'Number of samples: {len(training_df)}')

# 检查X和y的样本数量是否匹配
print(f'Number of features (X): {X.shape[0]}')
print(f'Number of labels (y): {y.shape[0]}')

# 训练随机森林模型
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 训练集上的预测
y_train_pred = model.predict(X_train)

# 测试集上的预测
y_test_pred = model.predict(X_test)

# 计算训练集的RMSE和R²
train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
train_r2 = r2_score(y_train, y_train_pred)

# 计算测试集的RMSE和R²
test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
test_r2 = r2_score(y_test, y_test_pred)

print(f'Training RMSE: {train_rmse}')
print(f'Training R²: {train_r2}')
print(f'Testing RMSE: {test_rmse}')
print(f'Testing R²: {test_r2}')
