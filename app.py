from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, StandardScaler
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error
from sklearn.decomposition import TruncatedSVD

# load data
data = pd.read_excel('data/modelling_data.xlsx')

data = data[data['exposure'] > 0.01]

# reset the index after filtering
data.reset_index(drop=True, inplace=True)

# define features and target columns
categorical_columns = ['annual_mileage', 'winter_tires', 'gender', 'location', 
                       'annual_income', 'ownership', 'occupation', 'credit_band', 
                       'marital_status', 'vehicle_value', 'car_model']
target_column = 'claimcount'
exposure_column = 'exposure'

# train test split
initial_X_train, X_test, initial_y_train, y_test = train_test_split(data.drop(target_column,axis=1), data[target_column], test_size=0.2, random_state=42)
initial_train_data = pd.concat([initial_X_train,initial_y_train],axis=1)

# cross-validation with 5 folds
kf = KFold(n_splits=5, shuffle=True, random_state=42)  
rmse_list = []

# cross-validation loop
for train_index, val_index in kf.split(initial_train_data):

    # split data into training and validation sets
    train_data, val_data = initial_train_data.iloc[train_index], initial_train_data.iloc[val_index]
    
    # separate exposure and claimcount for train set and validation set
    train_exposure = train_data[exposure_column]
    val_exposure = val_data[exposure_column]
    train_claimcount = train_data[target_column]
    val_claimcount = val_data[target_column]
    
    # one hot encode categorical columns
    encoder = OneHotEncoder()
    train_encoded = encoder.fit_transform(train_data[categorical_columns]).toarray()
    val_encoded = encoder.transform(val_data[categorical_columns]).toarray()
    
    # drop processed categorical columns from train and validation data
    train_data = train_data.drop(columns=categorical_columns + [exposure_column, target_column])
    val_data = val_data.drop(columns=categorical_columns + [exposure_column, target_column])
    
    # scale continuous features with min max
    scaler = MinMaxScaler()
    train_scaled = scaler.fit_transform(train_data)
    val_scaled = scaler.transform(val_data)
    
    # combine scaled and encoded features
    X_train = np.concatenate([train_scaled, train_encoded], axis=1)
    X_val = np.concatenate([val_scaled, val_encoded], axis=1)
    
    # add constant term (this slightly improves performance)
    X_train = sm.add_constant(X_train)
    X_val = sm.add_constant(X_val)

    svd = TruncatedSVD(n_components=15)  # this reduced the feature space to the given value (try tweaking this for better results)
    X_train = svd.fit_transform(X_train)
    X_val = svd.transform(X_val)

    
    # prepare target and offset
    y_train = train_claimcount.to_numpy()
    y_val = val_claimcount.to_numpy()

    # take log of offset (this is what research says to do)
    offset_train = np.log(train_exposure)
    offset_val = np.log(val_exposure)

    
    # fit model on training set
    model = sm.GLM(y_train, X_train, family=sm.families.Poisson(), offset=offset_train)
    result = model.fit(cov_type='HC3')
    
    # predict on validation set
    predicted_values = result.predict(X_val)
    
    # calculate RMSE for the fold
    mse = mean_squared_error(y_val, predicted_values)
    rmse = np.sqrt(mse)
    rmse_list.append(rmse)

# calculate and print average RMSE across folds
average_rmse = np.mean(rmse_list)
print(f'Cross-validated RMSE on Training Data: {average_rmse}')

# separate exposure
test_exposure = X_test[exposure_column]
# one hot encode categorical columns
test_encoded = encoder.transform(X_test[categorical_columns]).toarray()
# drop processed categorical columns from test data
X_test = X_test.drop(columns=categorical_columns + [exposure_column])
# scale continuous features with min max
test_scaled = scaler.transform(X_test)
# combine scaled and encoded features
X_test = np.concatenate([test_scaled, test_encoded], axis=1)
# add constant term (this slightly improves performance)
X_test = sm.add_constant(X_test)
X_test = svd.transform(X_test)
# prepare target and offset
y_test = y_test.to_numpy()
# take log of offset (this is what research says to do)
offset_test = np.log(test_exposure)
# predict on held out test set
predicted_values = result.predict(X_test)
# calculate RMSE for the fold
mse = mean_squared_error(y_test, predicted_values)
rmse = np.sqrt(mse)
print(f'RMSE on Test Data: {rmse}')