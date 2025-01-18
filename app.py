from sklearn.model_selection import KFold
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, StandardScaler
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error
from sklearn.decomposition import TruncatedSVD

# Load data
data = pd.read_excel('data/modelling_data.xlsx')

data = data[data['exposure'] > 0.01]

# Step 2: Reset the index after filtering
data.reset_index(drop=True, inplace=True)

# Define features and target columns
categorical_columns = ['annual_mileage', 'winter_tires', 'gender', 'location', 
                       'annual_income', 'ownership', 'occupation', 'credit_band', 
                       'marital_status', 'vehicle_value', 'car_model']
target_column = 'claimcount'
exposure_column = 'exposure'

# Step 3: Prepare for cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)  # 5-fold cross-validation
rmse_list = []

# Step 4: Cross-validation loop
for train_index, val_index in kf.split(data):
    # Split data into training and validation sets
    train_data, val_data = data.iloc[train_index], data.iloc[val_index]
    
    # Separate exposure and claimcount
    train_exposure = train_data[exposure_column]

    val_exposure = val_data[exposure_column]
    train_claimcount = train_data[target_column]
    val_claimcount = val_data[target_column]
    
    # Process categorical columns (fit encoder only on training data)
    encoder = OneHotEncoder()
    train_encoded = encoder.fit_transform(train_data[categorical_columns]).toarray()
    val_encoded = encoder.transform(val_data[categorical_columns]).toarray()
    
    # Drop processed categorical columns from train and validation data
    train_data = train_data.drop(columns=categorical_columns + [exposure_column, target_column])
    val_data = val_data.drop(columns=categorical_columns + [exposure_column, target_column])
    
    # Normalize continuous features (fit scaler only on training data)
    scaler = MinMaxScaler()
    train_scaled = scaler.fit_transform(train_data)
    val_scaled = scaler.transform(val_data)
    
    # Combine scaled and encoded features
    X_train = np.concatenate([train_scaled, train_encoded], axis=1)
    X_val = np.concatenate([val_scaled, val_encoded], axis=1)
    
    # Add constant term
    X_train = sm.add_constant(X_train)
    X_val = sm.add_constant(X_val)

    svd = TruncatedSVD(n_components=15)  # Adjust the number of components based on your needs
    X_train = svd.fit_transform(X_train)
    X_val = svd.transform(X_val)

    
    # Prepare target and offset
    y_train = train_claimcount.to_numpy()
    y_val = val_claimcount.to_numpy()

    offset_train = np.log(train_exposure)
    offset_val = np.log(val_exposure)

    
    # Fit model on training data
    model = sm.GLM(y_train, X_train, family=sm.families.Poisson(), offset=offset_train)
    result = model.fit(cov_type='HC3')
    
    # Predict on validation data
    predicted_values = result.predict(X_val)
    
    # Calculate RMSE for the fold
    mse = mean_squared_error(y_val, predicted_values)
    rmse = np.sqrt(mse)
    rmse_list.append(rmse)

# Step 5: Calculate and print average RMSE across folds
average_rmse = np.mean(rmse_list)
print(f'Cross-validated RMSE: {average_rmse}')
