import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_csv('Scooter demand prediction data.csv')

# Convert 'dteday' to datetime with correct format
df['dteday'] = pd.to_datetime(df['dteday'], dayfirst=True)
print(df.isnull().sum())
df = df.drop_duplicates()

# Exploratory Data Analysis (EDA)
# Plotting the distribution of the target variable 'cnt'
sns.histplot(df['cnt'], kde=True)
plt.title('Distribution of Scooter Demand (cnt)')
plt.show()

# Plotting correlation heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()

# Scatter plots for key variables against 'cnt'
sns.scatterplot(x='temp', y='cnt', data=df)
plt.title('Scooter Demand vs Temperature')
plt.show()

sns.scatterplot(x='hum', y='cnt', data=df)
plt.title('Scooter Demand vs Humidity')
plt.show()

sns.scatterplot(x='windspeed', y='cnt', data=df)
plt.title('Scooter Demand vs Windspeed')
plt.show()

# Boxplot for categorical variables
sns.boxplot(x='season', y='cnt', data=df)
plt.title('Scooter Demand by Season')
plt.show()

sns.boxplot(x='weathersit', y='cnt', data=df)
plt.title('Scooter Demand by Weather Situation')
plt.show()

# Feature Engineering
# Create 'day_of_week' feature from 'dteday'
df['day_of_week'] = df['dteday'].dt.dayofweek

# One-hot encoding for categorical variables
df = pd.get_dummies(df, columns=['season', 'weathersit', 'weekday', 'mnth'], drop_first=True)

# Dropping columns that are not needed or redundant
df = df.drop(columns=['instant', 'dteday', 'casual', 'registered'])

# Model Selection and Training
# Splitting the data
X = df.drop(columns=['cnt']) 
y = df['cnt']  
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Training Linear Regression model
lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)

# Training Random Forest Regressor model
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

# Evaluating models
print("Linear Regression R^2 Score:", r2_score(y_test, y_pred_lr))
print("Linear Regression RMSE:", mean_squared_error(y_test, y_pred_lr, squared=False))
print("Random Forest R^2 Score:", r2_score(y_test, y_pred_rf))
print("Random Forest RMSE:", mean_squared_error(y_test, y_pred_rf, squared=False))

# Feature importance from Random Forest
importances = rf.feature_importances_
features = X.columns
feature_importance_df = pd.DataFrame({'Feature': features, 'Importance': importances})
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

# Plotting feature importance
sns.barplot(x='Importance', y='Feature', data=feature_importance_df)
plt.title('Feature Importance')
plt.show()
