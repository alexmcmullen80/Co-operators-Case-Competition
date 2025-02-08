import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.metrics import mean_squared_error
import statsmodels.api as sm

# Load data
data = pd.read_excel('data/modelling_data.xlsx')
# eval_data = pd.read_excel('data/evaluation_data.xlsx')


# Define categorical and other columns
categorical_columns = ['annual_mileage', 'winter_tires', 'gender', 'location', 'deductible',
                       'annual_income', 'ownership', 'occupation', 'credit_band', 
                       'marital_status', 'vehicle_value', 'car_model']
target_column = 'claimcount'
exposure_column = 'exposure'

# Ensure the base level of each categorical variable is the one with the highest exposure
for col in categorical_columns:
    exposure_per_category = data.groupby(col)[exposure_column].sum()
    highest_exposure_category = exposure_per_category.idxmax()
    data[col] = pd.Categorical(
        data[col], 
        categories=[highest_exposure_category] + [cat for cat in data[col].unique() if cat != highest_exposure_category],
        ordered=True
    )

# Train-test split
initial_X_train, X_test, initial_y_train, y_test = train_test_split(
    data.drop(target_column, axis=1),
    data[target_column],
    test_size=0.2,
    random_state=42
)

# Train on all of modelling_data and test on all of evalation_data
# initial_y_train = data[target_column]
# initial_X_train = data.drop(target_column, axis=1)
# y_test = eval_data[target_column]
# X_test = eval_data.drop(target_column, axis=1)

# Prepare output
# df = pd.DataFrame(X_test, columns=['ROW_ID','exposure'])

# Prepare training data
X_train = initial_X_train.copy()
y_train = initial_y_train.values
X_test = X_test.copy()

# Add log-transformed exposure as a feature
X_train['log_exposure'] = np.log(X_train[exposure_column])
X_test['log_exposure'] = np.log(X_test[exposure_column])

# Separate numerical and categorical columns

numerical_columns = []
for col in X_train.columns:
    if col not in categorical_columns and col != exposure_column:
        numerical_columns.append(col)




# Scale numerical columns
scaler = MinMaxScaler()
X_train_scaled_numerical = scaler.fit_transform(X_train[numerical_columns])
X_test_scaled_numerical = scaler.transform(X_test[numerical_columns])

# One-hot encode categorical columns
encoder = OneHotEncoder(drop='first', sparse_output=False)  # Drop the base level for one-hot encoding. This is done to avoid multicollinearity, for example don't store Males=[1,0] and Females=[0,1], just store Female=[1]
X_train_encoded_categorical = encoder.fit_transform(X_train[categorical_columns])
X_test_encoded_categorical = encoder.transform(X_test[categorical_columns])

# Combine scaled numerical and encoded categorical features
X_train_prepared = np.hstack((X_train_scaled_numerical, X_train_encoded_categorical))
X_test_prepared = np.hstack((X_test_scaled_numerical, X_test_encoded_categorical))

#add an intercept so the model is not forced to pass through (0,0)
X_train_prepared = sm.add_constant(X_train_prepared)
X_test_prepared = sm.add_constant(X_test_prepared)

# Build model using poisson family
model = sm.GLM(y_train, X_train_prepared, family=sm.families.Poisson(), offset=X_train['log_exposure'].values)
result = model.fit()

# Use model to predict on the test set
offset_test = X_test['log_exposure'].values
y_pred = result.predict(X_test_prepared, offset=offset_test)

# Complete output
# df['prediction'] = y_pred
# df.to_excel('data/predictions.xlsx')

# Calculate RMSE
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f'RMSE on Test Data: {rmse}')