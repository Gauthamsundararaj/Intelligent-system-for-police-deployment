from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import datetime

app = Flask(__name__)
CORS(app)


# Load CSV data
try:
    data = pd.read_csv("crime_data.csv", parse_dates=["date"])
except Exception as e:
    data = pd.DataFrame()
    print("Error loading CSV:", e)

# Preprocess for trends
if not data.empty:
    data['hour'] = pd.to_datetime(data['time'], format='%H:%M').dt.hour
    data['day_of_week'] = data['date'].dt.day_name()
    data['month'] = data['date'].dt.month


def predict_crime(location, hour):
    filtered = data[(data['location'] == location) & (data['hour'] == hour)]
    if filtered.empty:
        return {'crime_risk': 'Low', 'crime_type': 'N/A'}
    crime_type = filtered['crime_type'].mode()[0]
    risk = "High" if len(filtered) > 5 else "Medium"
    return {'crime_risk': risk, 'crime_type': crime_type}


@app.route('/')
def home():
    return "🚓 Police Deployment System is running. Use /predict, /visualize, or /deploy-officers."


@app.route('/predict', methods=['POST'])
def predict():
    try:
        content = request.get_json()
        location = content['location']
        time_str = content['time']
        hour = int(datetime.datetime.strptime(time_str, "%H:%M").hour)
        result = predict_crime(location, hour)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 400


@app.route('/visualize', methods=['GET'])
def visualize():
    if data.empty:
        return jsonify({'error': 'No data loaded'}), 500
    crime_counts = data['location'].value_counts().to_dict()
    hourly_trends = data.groupby('hour').size().to_dict()
    return jsonify({
        'crime_by_location': crime_counts,
        'crime_by_hour': hourly_trends
    })


@app.route('/deploy-officers', methods=['GET'])
def deploy():
    if data.empty:
        return jsonify({'error': 'No data loaded'}), 500
    recent = data[data['date'] > pd.Timestamp.now() - pd.Timedelta(days=30)]
    top_locations = recent['location'].value_counts().nlargest(3).index.tolist()
    return jsonify({
        'recommended_locations': top_locations
    })


if __name__ == '__main__':
    app.run(debug=True)
