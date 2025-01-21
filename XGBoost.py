import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.metrics import mean_squared_error, make_scorer
import xgboost as xgb

# Define RMSE function and scorer
def rmse_score(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

rmse_scorer = make_scorer(rmse_score, greater_is_better=False)

# Define hyperparameter grid
param_grid = {
    'n_estimators': [50,100],
    'max_depth': [2, 3],
    'learning_rate': [0.01, 0.1],
    'subsample': [0.7, 0.9],
    'colsample_bytree': [0.8, 1.0],
    'gamma': [0],
    'reg_alpha': [1,2],
    'reg_lambda': [10,15],
}

# Load data
data = pd.read_excel('data/modelling_data.xlsx')

# Filter data with exposure > 0.01
data = data[data['exposure'] > 0.01]
data.reset_index(drop=True, inplace=True)

# Define features and target columns
categorical_columns = ['annual_mileage', 'winter_tires', 'gender', 'location', 
                       'annual_income', 'ownership', 'occupation', 'credit_band', 
                       'marital_status', 'vehicle_value', 'car_model']
target_column = 'claimcount'
exposure_column = 'exposure'

# Train-test split
initial_X_train, X_test, initial_y_train, y_test = train_test_split(
    data.drop(target_column, axis=1),
    data[target_column],
    test_size=0.2,
    random_state=42
)

# Preprocess categorical columns with OneHotEncoder
encoder = OneHotEncoder()
encoder.fit(initial_X_train[categorical_columns])

# Preprocess continuous features with MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(initial_X_train.drop(columns=categorical_columns + [exposure_column]))

# Prepare training data
train_encoded = encoder.transform(initial_X_train[categorical_columns]).toarray()
train_scaled = scaler.transform(initial_X_train.drop(columns=categorical_columns + [exposure_column]))
X_train_prepared = np.concatenate([train_scaled, train_encoded], axis=1)

# Add log-transformed exposure as an additional feature (without scaling)
X_train_prepared = np.hstack([X_train_prepared, np.log(initial_X_train[exposure_column]).values.reshape(-1, 1)])
y_train = initial_y_train.values

# Prepare test data
test_encoded = encoder.transform(X_test[categorical_columns]).toarray()
test_scaled = scaler.transform(X_test.drop(columns=categorical_columns + [exposure_column]))
X_test_prepared = np.concatenate([test_scaled, test_encoded], axis=1)

# Add log-transformed exposure as an additional feature (without scaling)
X_test_prepared = np.hstack([X_test_prepared, np.log(X_test[exposure_column]).values.reshape(-1, 1)])
y_test = y_test.values


# Define XGBoost model
xgb_model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)

# Grid search with 5-fold cross-validation
grid_search = GridSearchCV(
    estimator=xgb_model,
    param_grid=param_grid,
    scoring=rmse_scorer,
    cv=5,
    verbose=1,
    n_jobs=-1
)

# Fit grid search
grid_search.fit(X_train_prepared, y_train)

# Best hyperparameters and model
print("Best Hyperparameters:", grid_search.best_params_)
best_model = grid_search.best_estimator_

# Predict on test set
test_predictions = best_model.predict(X_test_prepared)

# Calculate RMSE on test data
test_rmse = rmse_score(y_test, test_predictions)
print(f'RMSE on Test Data: {test_rmse}')
