import requests
import time

# Read the audio file as binary data
with open('/Users/faraaz/Desktop/audio.mp3', 'rb') as audio_file:
    audio_data = audio_file.read()

print(f"Sending request with audio data of length: {len(audio_data)}")
start_time = time.time()

# Send the request
response = requests.post('http://localhost:3000/transcribe', 
                         json={'audio_data': list(audio_data)},
                         timeout=600)  # Set a 10-minute timeout

end_time = time.time()
print(f"Request completed in {end_time - start_time:.2f} seconds")

# Print the response
print("Response status code:", response.status_code)
print("Response content:", response.json())