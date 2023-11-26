from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import pandas as pd

# Assuming 'price' is the target variable
data = pd.read_csv("/content/Housing.csv")

# Map 'yes' and 'no' to 1 and 0 for relevant columns
data['mainroad'] = data['mainroad'].map({'yes': 1, 'no': 0})
data['guestroom'] = data['guestroom'].map({'yes': 1, 'no': 0})
data['basement'] = data['basement'].map({'yes': 1, 'no': 0})
data['hotwaterheating'] = data['hotwaterheating'].map({'yes': 1, 'no': 0})
data['airconditioning'] = data['airconditioning'].map({'yes': 1, 'no': 0})

# Assuming 'price' is the target variable
X = data.drop('price', axis=1)
y = data['price']

# Convert categorical variables to numerical using one-hot encoding
X = pd.get_dummies(X, drop_first=True)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Random Forest Regressor
rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)

# Fit the model on the training data
rf_regressor.fit(X_train, y_train)

# Make predictions on the testing data
y_pred = rf_regressor.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

# Example: Predicting the price for a new house
new_house_features = pd.DataFrame({
    'area': [6000],
    'bedrooms': [3],
    'bathrooms': [2],
    'stories': [2],
    'mainroad': [1],
    'guestroom': [0],
    'basement': [1],
    'hotwaterheating': [0],
    'airconditioning': [1]
})

# Map 'yes' and 'no' to 1 and 0 in the new_house_features
new_house_features['mainroad'] = new_house_features['mainroad'].map({1: 1, 0: 0})
new_house_features['guestroom'] = new_house_features['guestroom'].map({1: 1, 0: 0})
new_house_features['basement'] = new_house_features['basement'].map({1: 1, 0: 0})
new_house_features['hotwaterheating'] = new_house_features['hotwaterheating'].map({1: 1, 0: 0})
new_house_features['airconditioning'] = new_house_features['airconditioning'].map({1: 1, 0: 0})

# Convert categorical variables to numerical using one-hot encoding
new_house_features = pd.get_dummies(new_house_features, drop_first=True)

# Ensure the columns are in the same order as in X_train
new_house_features = new_house_features.reindex(columns=X_train.columns, fill_value=0)

# Predict the price for the new house
predicted_price = rf_regressor.predict(new_house_features)
print(f'Predicted Price for the new house: {predicted_price[0]}')
