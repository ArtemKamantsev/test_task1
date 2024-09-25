from pickle import dump, load, HIGHEST_PROTOCOL

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler

data_path = '../data/train.csv'
model_save_path = '../models/random_forest_regressor.pickle'

# Read the data
df = pd.read_csv(data_path)
X_train, y_train = df.iloc[:, [6, 7]], df.target

# train the model
pipe = make_pipeline(MinMaxScaler(),
                     # suboptimal params are chosen to reduce the file size (file size with optimal params ~9GB)
                     RandomForestRegressor(n_jobs=-1, random_state=42, n_estimators=50))
pipe.fit(X_train, y_train)
print('Model trained')

# save the model
with open(model_save_path, "wb") as f:
    # It would be better to use ONNX for model persistence because pickle is unsafe,
    # but I had issues with saving to ONNX this model
    dump(pipe, f, HIGHEST_PROTOCOL)

# Check the model
with open(model_save_path, "rb") as f:
    pipe_restored = load(f)
print('Model saved')

prediction_original = pipe.predict(X_train)
prediction_restored = pipe_restored.predict(X_train)
discrepancy = (prediction_original - prediction_restored) ** 2

print(f'Max discrepancy: {discrepancy.max()}')
