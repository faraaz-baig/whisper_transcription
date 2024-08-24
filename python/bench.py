import requests
import time
import statistics
import concurrent.futures

def send_request(audio_data):
    start_time = time.time()
    response = requests.post('http://localhost:3000/transcribe', 
                             json={'audio_data': list(audio_data)})
    end_time = time.time()
    return end_time - start_time, response.json()

def run_benchmark(audio_file_path, num_requests=10, concurrent=False):
    with open(audio_file_path, 'rb') as audio_file:
        audio_data = audio_file.read()

    times = []
    
    if concurrent:
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_requests) as executor:
            futures = [executor.submit(send_request, audio_data) for _ in range(num_requests)]
            for future in concurrent.futures.as_completed(futures):
                time_taken, _ = future.result()
                times.append(time_taken)
    else:
        for _ in range(num_requests):
            time_taken, _ = send_request(audio_data)
            times.append(time_taken)

    return times

def print_stats(times):
    print(f"Number of requests: {len(times)}")
    print(f"Total time: {sum(times):.2f} seconds")
    print(f"Average time per request: {statistics.mean(times):.2f} seconds")
    print(f"Median time: {statistics.median(times):.2f} seconds")
    print(f"Min time: {min(times):.2f} seconds")
    print(f"Max time: {max(times):.2f} seconds")
    print(f"Standard deviation: {statistics.stdev(times):.2f} seconds")

if __name__ == "__main__":
    audio_file_path = '/Users/faraaz/Desktop/audio.mp3'  # Replace with your audio file path
    num_requests = 10  # Number of requests to send

    print("Running sequential benchmark...")
    sequential_times = run_benchmark(audio_file_path, num_requests, concurrent=False)
    print("\nSequential Benchmark Results:")
    print_stats(sequential_times)

    print("\nRunning concurrent benchmark...")
    concurrent_times = run_benchmark(audio_file_path, num_requests, concurrent=True)
    print("\nConcurrent Benchmark Results:")
    print_stats(concurrent_times)