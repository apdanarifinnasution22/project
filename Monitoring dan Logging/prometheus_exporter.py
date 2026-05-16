from prometheus_client import start_http_server, Counter
import time

REQUEST_COUNT = Counter('request_count', 'Total Request')

if __name__ == '__main__':
    start_http_server(8000)

    while True:
        REQUEST_COUNT.inc()
        print("Monitoring berjalan...")
        time.sleep(5)