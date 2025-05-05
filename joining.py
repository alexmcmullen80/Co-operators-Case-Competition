import pandas as pd
data = pd.read_excel('data/modelling_data.xlsx')
data2 = pd.read_excel('data/final_round_dataset.xlsx')


result = pd.merge(data, data2, on=['annual_mileage', 'winter_tires', 'gender', 'location', 'deductible',
                       'annual_income', 'ownership', 'occupation', 'credit_band', 'claims_history', 'years_driving',
                       'loyalty', 'marital_status', 'age_of_insured', 'vehicle_value', 'car_year','car_model'], how='inner')



result.to_excel('data/joined_data.xlsx', index=False)
