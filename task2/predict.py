from pickle import load

import pandas as pd

model_save_path = '../models/random_forest_regressor.pickle'
prediction_data_path = '../data/hidden_test.csv'
prediction_output_path = '../data/hidden_test_prediction.csv'

df = pd.read_csv(prediction_data_path)
X_test = df.iloc[:, [6, 7]]

with open(model_save_path, "rb") as f:
    pipe = load(f)

prediction = pipe.predict(X_test)

df_result = pd.DataFrame({'target_predicted': prediction})
df_result.to_csv(prediction_output_path, index=False)
