from flask import Flask, request, jsonify, render_template, send_file
import pandas as pd
import os
import base64
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest

app = Flask(__name__)

CSV_FILE = 'IOT_requests/dataset-100000_with_anomalies.csv'
# Use Agg backend for matplotlib
plt.switch_backend('Agg')
# Initialize a global variable to keep track of the last anomaly check time
last_check_time = None
@app.route('/')
def index():

    return render_template('index.html')

@app.route('/', methods=['POST'])
def process_data():
    global last_check_time

    data = request.json

    required_columns = ['Timestamp', 'SourceIP', 'DestinationPort', 'Protocol', 'BytesTransferred', 'EventType', 'ResponseTime', 'Status']

    if not all(col in data for col in required_columns):
        return jsonify({"error": "All parameters are required."}), 400
    
    # Convert the 'Timestamp' field to the desired format "M/D/YYYY H:MM"
    data['Timestamp'] = pd.to_datetime(data['Timestamp']).strftime('%-m/%-d/%Y %-H:%M')
    
    # Convert incoming data to DataFrame
    df = pd.DataFrame([data])

    # Save the incoming data to the CSV file
    if os.path.exists(CSV_FILE):
        df.to_csv(CSV_FILE, mode='a', header=False, index=False)
    else:
        df.to_csv(CSV_FILE, mode='w', header=True, index=False)


    anomalies, image_path=check_for_anomalies()
    
    # Convert image to base64
    if image_path:
        with open(image_path, "rb") as img_file:
            encoded_img_data = base64.b64encode(img_file.read()).decode('utf-8')
    else:
        encoded_img_data = None      

    return jsonify({
        "anomalies": anomalies,
        "img_data": encoded_img_data
    })

def check_for_anomalies(tail=15):
    anomaly_messages = []
    if os.path.exists(CSV_FILE):
        df = pd.read_csv(CSV_FILE)

        # Step 2: Pre-process the dataset
        df['Timestamp'] = pd.to_datetime(df['Timestamp'])
        df.set_index('Timestamp', inplace=True)

        # Resample the data to get the request count per minute per IOT Devices
        request_count = df.groupby('IOT Devices').resample('T').size().reset_index(name='Request_Count')

        # Initialize a list to store the anomaly detection results for each IOT Devices
        anomaly_results = []

        # Loop through each unique IOT Devices
        for event_type in request_count['IOT Devices'].unique():
            # Filter data for the current IOT Devices
            event_data = request_count[request_count['IOT Devices'] == event_type]
            
            # Split the data into two parts: the main dataset and the last 15 minutes
            last_15_min = event_data.tail(tail).copy()  # Create a copy to avoid the warning
            train_data = event_data.iloc[:-tail].copy()  # Create a copy to avoid the warning

            # Feature engineering: Create additional features for isolation forest
            train_data.loc[:, 'MinuteOfDay'] = train_data['Timestamp'].dt.hour * 60 + train_data['Timestamp'].dt.minute
            train_data.loc[:, 'DayOfWeek'] = train_data['Timestamp'].dt.weekday

            # Train Isolation Forest on the main dataset
            clf = IsolationForest(contamination=0.01, random_state=42)
            clf.fit(train_data[['Request_Count', 'MinuteOfDay', 'DayOfWeek']])

            # Apply the trained model to the last 15 minutes of data
            last_15_min.loc[:, 'MinuteOfDay'] = last_15_min['Timestamp'].dt.hour * 60 + last_15_min['Timestamp'].dt.minute
            last_15_min.loc[:, 'DayOfWeek'] = last_15_min['Timestamp'].dt.weekday
            last_15_min.loc[:, 'Anomaly'] = clf.predict(last_15_min[['Request_Count', 'MinuteOfDay', 'DayOfWeek']])
            # Check for anomalies in the last 15 minutes
            anomalies_exist = last_15_min['Anomaly'].eq(-1).any()
            # Store the result for this IOT Devices
            anomaly_results.append((event_type, anomalies_exist, last_15_min))

            # Generate an anomaly message
            if anomalies_exist:
                message = f"Anomalies detected in the last 15 minutes for IOT Devices: {event_type}."
            else:
                message = f"No anomalies detected in the last 15 minutes for IOT Devices: {event_type}."

            # Store the result message
            print(message)
            anomaly_messages.append(message)

        # Visualization: Plot all IOT Devices on the same diagram
        plt.figure(figsize=(15, 6))

        for event_type, anomalies_exist, last_15_min in anomaly_results:
            plt.plot(last_15_min['Timestamp'], last_15_min['Request_Count'], label=f'{event_type} Request Count')
            plt.scatter(last_15_min[last_15_min['Anomaly'] == -1]['Timestamp'], 
                        last_15_min[last_15_min['Anomaly'] == -1]['Request_Count'], 
                        label=f'{event_type} Anomalies', marker='x')

        plt.title('Anomaly Detection for All IOT Devices')
        plt.xlabel('Timestamp')
        plt.ylabel('Request Count')
        plt.legend()

        # Save the plot as an image
        image_path = 'static/anomaly_detection_plot.png'
        plt.savefig(image_path)
        plt.close()

        return anomaly_messages, image_path
    else:
        return ["No data available for anomaly detection."], None


if __name__ == '__main__':
    app.run(debug=True)