import requests

# Read the audio file as binary data
with open('/Users/faraaz/Desktop/audio.mp3', 'rb') as audio_file:
    audio_data = audio_file.read()

# Send the request
response = requests.post('http://localhost:3000/transcribe', 
                         json={'audio_data': list(audio_data)})

# Print the response
print(response.json())