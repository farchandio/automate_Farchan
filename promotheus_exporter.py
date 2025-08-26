import time
import requests
import json
from flask import Flask, request, Response
from prometheus_client import Counter, Histogram, generate_latest

# --- 1. Definisikan Metrik yang Akan Di-monitor ---
# Metrik 1: Menghitung total request yang masuk
REQUEST_COUNT = Counter(
    'http_requests_total',
    'Total HTTP Requests',
    ['method', 'endpoint', 'status_code']
)

# Metrik 2: Mengukur latensi atau waktu respon request
REQUEST_LATENCY = Histogram(
    'request_latency_seconds',
    'Request Latency',
    ['endpoint']
)

# Metrik 3: Menghitung hasil prediksi (custom metric)
PREDICTION_RESULTS = Counter(
    'prediction_results_total',
    'Total Prediction Results',
    ['outcome'] # 'survived' or 'not_survived'
)

# --- 2. Buat Aplikasi Flask sebagai Jembatan ---
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    start_time = time.time()
    
    # Ambil data dari request
    json_data = request.get_json()
    
    # Teruskan request ke model MLflow yang sedang berjalan di port 5001
    try:
        response = requests.post(
            'http://127.0.0.1:5001/invocations',
            json=json_data,
            headers={'Content-Type': 'application/json'}
        )
        prediction = response.json()
        status_code = response.status_code
        
        # Catat hasil prediksi
        outcome = 'survived' if prediction.get('predictions', [0])[0] == 1 else 'not_survived'
        PREDICTION_RESULTS.labels(outcome=outcome).inc()

    except requests.exceptions.RequestException as e:
        prediction = {"error": str(e)}
        status_code = 500

    # Catat metrik request count
    REQUEST_COUNT.labels(method='POST', endpoint='/predict', status_code=status_code).inc()
    
    # Catat metrik latensi
    latency = time.time() - start_time
    REQUEST_LATENCY.labels(endpoint='/predict').observe(latency)
    
    return Response(json.dumps(prediction), status=status_code, mimetype='application/json')

@app.route('/metrics')
def metrics():
    # Endpoint ini akan di-scrape oleh Prometheus
    return Response(generate_latest(), mimetype='text/plain')

if __name__ == '__main__':
    # Jalankan exporter di port 8000
    app.run(host='0.0.0.0', port=8000)