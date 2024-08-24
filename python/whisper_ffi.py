import sys
import tempfile
import os

try:
    from faster_whisper import WhisperModel
except ImportError:
    print("Error: The 'faster-whisper' package is not installed.")
    print("Please install it using: pip install faster-whisper")
    sys.exit(1)

class WhisperTranscriber:
    def __init__(self, model_size="large-v3", device="cuda", compute_type="float16"):
        try:
            self.model = WhisperModel(model_size, device=device, compute_type=compute_type)
        except Exception as e:
            print(f"Error initializing WhisperModel: {e}")
            raise

    def transcribe(self, audio_data):
        try:
            # Create a temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
                temp_audio.write(audio_data)
                temp_audio_path = temp_audio.name

            # Transcribe using the temporary file path
            segments, info = self.model.transcribe(temp_audio_path)
            transcription = " ".join(segment.text for segment in segments)

            # Delete the temporary file
            os.unlink(temp_audio_path)

            return transcription, info.language, info.language_probability
        except Exception as e:
            print(f"Error during transcription: {e}")
            raise

transcriber = None

def initialize_transcriber():
    global transcriber
    try:
        transcriber = WhisperTranscriber()
        print("Transcriber initialized successfully.")
    except Exception as e:
        print(f"Failed to initialize transcriber: {e}")
        raise

def transcribe_audio(audio_data):
    if transcriber is None:
        raise RuntimeError("Transcriber not initialized")
    return transcriber.transcribe(audio_data)

if __name__ == "__main__":
    print("Testing WhisperTranscriber initialization...")
    initialize_transcriber()
    print("Initialization test complete.")