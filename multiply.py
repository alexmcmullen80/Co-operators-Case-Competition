import pandas as pd
severity = pd.read_excel('data/severity.xlsx')
predictions = pd.read_excel('data/predictions_on_train.xlsx')
# predictions.insert(0, 'ID', range(1, len(predictions) + 1))
predictions['loss'] = severity['severity'] * predictions['prediction']

n = int(len(predictions) * 0.05)

# Get the top 5% highest losses
top_5_percent_loss = predictions.nlargest(n, 'loss')

# Display the result
#top_5_percent_loss.to_excel('data/top_5_percent_loss.xlsx', index=False)
# predictions.to_excel('data/predictions_with_loss.xlsx', index=False)