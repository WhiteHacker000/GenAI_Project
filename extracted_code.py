import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

# from google.colab import drive
# drive.mount('/content/drive')

df = pd.read_csv('EVChargingStationUsage.csv')

df.columns

df['Station Name'].nunique()
df['Station Name'].value_counts(normalize=True).head()

print("Unique Stations:", df['Station Name'].nunique())
df['Station Name'].value_counts().head(10)

df['Start Date'] = pd.to_datetime(df['Start Date'])
df['Hour'] = df['Start Date'].dt.hour
df['Weekday'] = df['Start Date'].dt.weekday


print("Unique Hours:", df['Hour'].nunique())
print("Unique Weekdays:", df['Weekday'].nunique())

df['Hour'].nunique()
df['Weekday'].nunique()

df['Energy (kWh)'].describe()

import matplotlib.pyplot as plt
plt.hist(df['Energy (kWh)'], bins=30)
plt.show()

df_grouped = df.groupby(['Station Name', 'Hour'])['Energy (kWh)'].sum().reset_index()

print("Stations:", df['Station Name'].nunique())
print("Hours:", df['Hour'].nunique())
print("Weekdays:", df['Weekday'].nunique())

df['City'].value_counts(normalize=True)

df.info()

df.shape

##these are unnecessary columns

df = df.drop(columns=[
    'MAC Address',
    'User ID',
    'Plug In Event Id',
    'Driver Postal Code',
    'System S/N',
    'Model Number',
    'Start Time Zone',
    'End Time Zone',
    'Transaction Date (Pacific Time)',
    'Address 1',
    'Postal Code',
    'Currency'
])

##these columns have the too many missing values

df = df.drop(columns=['EVSE ID', 'County'])

##removing non-zero energy sessions

df = df[df['Energy (kWh)'] > 0] #non-zero doesn't represent demand

df['Charging Time (hh:mm:ss)'] = pd.to_timedelta(df['Charging Time (hh:mm:ss)'])
df['Charging_Time_Minutes'] = df['Charging Time (hh:mm:ss)'].dt.total_seconds() / 60

df['Total Duration (hh:mm:ss)'] = pd.to_timedelta(df['Total Duration (hh:mm:ss)'])
df['Total_Duration_Minutes'] = df['Total Duration (hh:mm:ss)'].dt.total_seconds() / 60

df = df[[
    'Station Name',
    'City',
    'State/Province',
    'Country',
    'Latitude',
    'Longitude',
    'Port Type',
    'Plug Type',
    'Fee',
    'Energy (kWh)',
    'Charging_Time_Minutes',
    'Total_Duration_Minutes',
    'Hour',
    'Weekday'
]]

df.info()

df.shape

df_grouped = df.groupby(['Station Name', 'Hour', 'Weekday'])['Energy (kWh)'].sum().reset_index()

df.shape

df.isnull().sum().sum()

df.isnull().sum()

df['Port Type'] = df['Port Type'].fillna(df['Port Type'].mode()[0])

df.isnull().sum()

df['Port Type'].value_counts()

## we drop this becuase of the imbalance

df = df.drop(columns=['Port Type'])

df_grouped.head()

df_grouped.shape

df_grouped['Peak_Hour'] = df_grouped['Hour'].apply(
    lambda x: 1 if (6 <= x <= 10) or (17 <= x <= 21) else 0
)

df_grouped['Is_Weekend'] = df_grouped['Weekday'].apply(
    lambda x: 1 if x >= 5 else 0
)

station_avg = df_grouped.groupby('Station Name')['Energy (kWh)'].mean()
df_grouped['Station_Avg_Load'] = df_grouped['Station Name'].map(station_avg)

df_grouped['Hour_Weekend_Interaction'] = df_grouped['Hour'] * df_grouped['Is_Weekend']

df_grouped.columns

df_grouped = pd.get_dummies(df_grouped, columns=['Station Name'], drop_first=True)

## Defined Feature and Target

X = df_grouped.drop(columns=['Energy (kWh)'])
y = df_grouped['Energy (kWh)']

##Train-Test Spit

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Peak hour indicator
df_grouped['Peak_Hour'] = df_grouped['Hour'].apply(
    lambda x: 1 if (6 <= x <= 10) or (17 <= x <= 21) else 0
)

# Weekend indicator
df_grouped['Is_Weekend'] = df_grouped['Weekday'].apply(
    lambda x: 1 if x >= 5 else 0
)


# Interaction feature
df_grouped['Hour_Weekend_Interaction'] = df_grouped['Hour'] * df_grouped['Is_Weekend']

df_grouped.head()

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

lr = LinearRegression()
lr.fit(X_train, y_train)

lr_pred = lr.predict(X_test)

def evaluate_model(name, y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    
    print(f"Model: {name}")
    print(f"MAE  : {mae:.4f}")
    print(f"MSE  : {mse:.4f}")
    print(f"RMSE : {rmse:.4f}")
    print(f"R2   : {r2:.4f}")

evaluate_model("Linear Regression", y_test, lr_pred)

rf = RandomForestRegressor(
    n_estimators=200,
    random_state=42
)

rf.fit(X_train, y_train)

rf_pred = rf.predict(X_test)

gb = GradientBoostingRegressor(
    n_estimators=200,
    learning_rate=0.1,
    random_state=42
)

gb.fit(X_train, y_train)

gb_pred = gb.predict(X_test)

def evaluate_model(name, y_test, y_pred):
    print(f"\n{name}")
    print("MAE:", mean_absolute_error(y_test, y_pred))
    print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))
    print("R2:", r2_score(y_test, y_pred))

evaluate_model("Random Forest", y_test, rf_pred)


evaluate_model("Gradient Boosting", y_test, gb_pred)

