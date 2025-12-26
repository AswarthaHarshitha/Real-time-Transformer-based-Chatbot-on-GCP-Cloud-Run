import time
import httpx
import statistics

URL = "http://localhost:8080/chat"  # replace with service URL

def measure(n=50):
    latencies = []
    client = httpx.Client(timeout=10.0)
    for i in range(n):
        start = time.time()
        r = client.post(URL, json={"prompt": "Hello, how are you?", "max_new_tokens": 32})
        dt = (time.time() - start) * 1000
        latencies.append(dt)
        print(f"{i+1}/{n}: {dt:.1f} ms, status={r.status_code}")
    print("\nSummary:")
    print(f"mean: {statistics.mean(latencies):.1f} ms")
    print(f"p50: {statistics.median(latencies):.1f} ms")
    print(f"p95: {sorted(latencies)[int(len(latencies)*0.95)-1]:.1f} ms")

if __name__ == '__main__':
    measure()
