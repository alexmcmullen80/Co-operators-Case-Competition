import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error




# Load data
data = pd.read_excel('data/modelling_data.xlsx')

# Step 1: Filter rows with exposure >= 0.0001
data = data[data['exposure'] > 1e-3]

# Step 2: Reset the index after filtering
data.reset_index(drop=True, inplace=True)

# Step 3: One-hot encode categorical columns
encoder = OneHotEncoder()
encoded_array = encoder.fit_transform(
    data[['annual_mileage', 'winter_tires', 'gender', 'location', 
          'annual_income', 'ownership', 'occupation', 'credit_band', 
          'marital_status', 'vehicle_value', 'car_model']]).toarray()
#encoded_df = pd.DataFrame(encoded_array, columns=encoder.get_feature_names_out())

# Step 4: Drop original categorical columns
data.drop(
    ['annual_mileage', 'winter_tires', 'gender', 'location', 'annual_income', 
     'ownership', 'occupation', 'credit_band', 'marital_status', 
     'vehicle_value', 'car_model'], 
    axis=1, 
    inplace=True
)

# Step 5: Separate and process 'exposure' and 'claimcount'
new_data = data[['exposure', 'claimcount']].copy()
new_data['log_exposure'] = np.log(new_data['exposure'])

# Step 6: Remove exposure and claimcount from the main data for scaling
data.drop(['exposure', 'claimcount'], axis=1, inplace=True)

# Step 7: Normalize continuous features
scaler = MinMaxScaler()
data_array = scaler.fit_transform(data)

# Step 8: Combine normalized data with encoded features
data_array = np.concatenate((data_array, encoded_array), axis=1)

# Step 9: Extract the target variable and the offset
X = data_array
y = new_data['claimcount'].to_numpy()
offset = new_data['log_exposure'].to_numpy()

# Step 10: Add a constant term for the model
X = sm.add_constant(X)

# Step 11: Fit a Poisson regression model
model = sm.GLM(y, X, family=sm.families.Poisson(), offset=offset)
result = model.fit()

# # Step 12: Print model summary
# print(result.summary())

# Assuming 'result' is the fitted GLM model
predicted_values = result.predict(X)  # X is the feature matrix used for fitting the model


# Assuming 'y' is the actual target values and 'predicted_values' are the predicted values from the model

# Calculate Mean Squared Error (MSE)
mse = mean_squared_error(y, predicted_values)

# Calculate RMSE by taking the square root of MSE
rmse = np.sqrt(mse)

# Print the RMSE
print(f'RMSE: {rmse}')



