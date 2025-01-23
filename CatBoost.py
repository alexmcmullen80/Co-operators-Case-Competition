import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, make_scorer
from catboost import CatBoostRegressor

# Define RMSE function and scorer
def rmse_score(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

rmse_scorer = make_scorer(rmse_score, greater_is_better=False)

# Load data
data = pd.read_excel('data/modelling_data.xlsx')

# Filter data with exposure > 0.01 (i believe gives worse rmse)
# data = data[data['exposure'] > 0.01]
# data.reset_index(drop=True, inplace=True)

# group ages (makes a very slight difference) (you also must add 'age_group' to categorical_columns)
# data.loc[data['age_of_insured']<=19, 'age_group'] = 'a'
# data.loc[data['age_of_insured'].between(20,29), 'age_group'] = 'b'
# data.loc[data['age_of_insured'].between(30,39), 'age_group'] = 'bb'
# data.loc[data['age_of_insured'].between(40,49), 'age_group'] = 'c'
# data.loc[data['age_of_insured'].between(50,59), 'age_group'] = 'd'
# data.loc[data['age_of_insured'].between(60,69), 'age_group'] = 'e'
# data.loc[data['age_of_insured']>69, 'age_group'] = 'f'
# data = data.drop(columns=['age_of_insured'])

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

# Prepare training data
X_train = initial_X_train.copy()
y_train = initial_y_train.values
X_test = X_test.copy()

# Add log-transformed exposure as a feature
X_train['log_exposure'] = np.log(X_train[exposure_column])
X_test['log_exposure'] = np.log(X_test[exposure_column])

# Define categorical feature indices for CatBoost
categorical_indices = [X_train.columns.get_loc(col) for col in categorical_columns]

# # Define CatBoost hyperparameter grid
# param_grid = {
#     'iterations': [500,600],
#     'depth': [4],
#     'learning_rate': [0.001, 0.01],
#     'l2_leaf_reg': [3, 4],
#     'subsample': [0.9]
# }

# # Define CatBoost model
# catboost_model = CatBoostRegressor(
#     loss_function='RMSE',
#     cat_features=categorical_indices,
#     random_seed=42,
#     verbose=0
# )

# # Grid search with cross-validation
# grid_search = GridSearchCV(
#     estimator=catboost_model,
#     param_grid=param_grid,
#     scoring=rmse_scorer,
#     cv=2,
#     verbose=1,
#     n_jobs=-1
# )

tuned_model = CatBoostRegressor(
    depth=4,  # Test slightly deeper trees
    iterations=500,  # More iterations for potential improvement
    l2_leaf_reg=4,  # Slightly stronger regularization
    learning_rate=0.01,  # Lower learning rate with more iterations
    subsample=0.9,  # Introduce stochasticity
    loss_function='RMSE',
    cat_features=categorical_indices,
    random_seed=42
)

# # Fit grid search
# grid_search.fit(X_train, y_train)

# # Best hyperparameters and model
# print("Best Hyperparameters:", grid_search.best_params_)
# best_model = grid_search.best_estimator_
best_model = tuned_model.fit(X_train, y_train)

# Predict on test set
test_predictions = best_model.predict(X_test)

# Calculate RMSE on test data
test_rmse = rmse_score(y_test, test_predictions)
print(f'RMSE on Test Data: {test_rmse}')
