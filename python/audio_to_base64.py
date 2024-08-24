import base64
import sys

def audio_to_base64(file_path):
    with open(file_path, "rb") as audio_file:
        audio_content = audio_file.read()
        return base64.b64encode(audio_content).decode('utf-8')

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python audio_to_base64.py <path_to_audio_file>")
        sys.exit(1)
    
    audio_path = sys.argv[1]
    base64_audio = audio_to_base64(audio_path)
    print(base64_audio)