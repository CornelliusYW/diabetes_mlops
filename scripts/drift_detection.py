import pandas as pd
from evidently.metric_preset import DataDriftPreset
from evidently.report import Report

# Load your reference (training) and production (new) data
reference_data = pd.read_csv('data/reference_data.csv')
drift_data = pd.read_csv('data/drift_data.csv')

data_drift_report = Report(metrics=[
    DataDriftPreset()
])

data_drift_report.run(reference_data=reference_data, current_data=drift_data, column_mapping=None)

report_json = data_drift_report.as_dict()
drift_detected = report_json['metrics'][0]['result']['dataset_drift']

if drift_detected:
    print("Data drift detected. Retraining the model.")
    with open('drift_detected.txt', 'w') as f:
        f.write('drift_detected')
else:
    print("No data drift detected.")
    with open('drift_detected.txt', 'w') as f:
        f.write('no_drift')
