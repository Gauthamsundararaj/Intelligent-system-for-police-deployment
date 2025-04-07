import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv('crime_data.csv')
df['Date'] = pd.to_datetime(df['Date'])
df['Month'] = df['Date'].dt.month
df['Hour'] = df['Date'].dt.hour

le = LabelEncoder()
df['Crime_Type'] = le.fit_transform(df['Crime_Type'])

X = df[['Month', 'Hour', 'Location_Lat', 'Location_Lon']]
y = df['Crime_Type']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

joblib.dump(model, 'crime_model.pkl')
joblib.dump(le, 'label_encoder.pkl')

print("Model trained and saved!")
