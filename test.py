import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest

# Step 1: Load the dataset
df = pd.read_csv('IOT_requests/dataset-100000_with_anomalies.csv')

# Step 2: Pre-process the dataset
df['Timestamp'] = pd.to_datetime(df['Timestamp'])
df.set_index('Timestamp', inplace=True)

# Resample the data to get the request count per minute per IOT Devices
request_count = df.groupby('IOT Devices').resample('T').size().reset_index(name='Request_Count')
print(request_count.sort_values(by=['Request_Count']))
# Initialize a list to store the anomaly detection results for each IOT Devices
anomaly_results = []

# Loop through each unique IOT Devices
for event_type in request_count['IOT Devices'].unique():
    # Filter data for the current IOT Devices
    event_data = request_count[request_count['IOT Devices'] == event_type]
    
    # Split the data into two parts: the main dataset and the last 15 minutes
    last_15_min = event_data.tail(15).copy()  # Create a copy to avoid the warning
    train_data = event_data.iloc[:-15].copy()  # Create a copy to avoid the warning

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

    # Print result for the current IOT Devices
    if anomalies_exist:
        print(f"Anomalies detected in the last 15 minutes for IOT Devices: {event_type}.")
    else:
        print(f"No anomalies detected in the last 15 minutes for IOT Devices: {event_type}.")

# Visualization: Plot all IOT Devicess on the same diagram
plt.figure(figsize=(15, 6))

for event_type, anomalies_exist, last_15_min in anomaly_results:
    plt.plot(last_15_min['Timestamp'], last_15_min['Request_Count'], label=f'{event_type} Request Count')
    plt.scatter(last_15_min[last_15_min['Anomaly'] == -1]['Timestamp'], 
                last_15_min[last_15_min['Anomaly'] == -1]['Request_Count'], 
                label=f'{event_type} Anomalies', marker='x')

plt.title('Anomaly Detection for All IOT Devicess')
plt.xlabel('Timestamp')
plt.ylabel('Request Count')
plt.legend()
plt.show()

image_path = 'static/anomaly_detection_plot.png'
plt.savefig(image_path)
plt.close()
