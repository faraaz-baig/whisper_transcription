import sys
import site
import os
from typing import Union
import io
import time
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

logger.info(f"Python version: {sys.version}")
logger.info(f"Python executable: {sys.executable}")
logger.info(f"sys.path: {sys.path}")
logger.info(f"site-packages: {site.getsitepackages()}")
logger.info(f"Current working directory: {os.getcwd()}")

try:
    logger.info("Attempting to import faster_whisper...")
    from faster_whisper import WhisperModel
    logger.info("WhisperModel imported successfully")
except ImportError as e:
    logger.error(f"Error importing faster_whisper: {e}")
    logger.error("The 'faster-whisper' package is not installed or there's an issue with the installation.")
    logger.error("Please install it using: uv pip install faster-whisper")
    logger.error("If the issue persists, try reinstalling with: uv pip install --force-reinstall faster-whisper")
    raise

class WhisperTranscriber:
    def __init__(self, model_size="medium", device="cpu", compute_type="int8"):
        try:
            logger.info(f"Initializing WhisperModel with: model_size={model_size}, device={device}, compute_type={compute_type}")
            start_time = time.time()
            self.model = WhisperModel(model_size, device=device, compute_type=compute_type)
            end_time = time.time()
            logger.info(f"WhisperTranscriber initialized successfully in {end_time - start_time:.2f} seconds")
        except Exception as e:
            logger.error(f"Error initializing WhisperModel: {e}")
            raise

    def transcribe(self, audio_data: Union[str, bytes]) -> tuple:
        try:
            logger.info(f"Starting transcription of audio data (type: {type(audio_data)})")
            start_time = time.time()
            
            if isinstance(audio_data, bytes):
                audio_data = io.BytesIO(audio_data)
            
            segments, info = self.model.transcribe(audio_data)
            
            all_segments = list(segments)
            transcription = " ".join(segment.text for segment in all_segments)

            end_time = time.time()
            logger.info(f"Transcription completed in {end_time - start_time:.2f} seconds")

            return transcription, info.language, info.language_probability, all_segments
        except Exception as e:
            logger.error(f"Error during transcription: {e}")
            raise

transcriber = None

def initialize_transcriber(model_size="medium", device="cpu", compute_type="int8"):
    global transcriber
    try:
        transcriber = WhisperTranscriber(model_size, device, compute_type)
        logger.info("Transcriber initialized successfully.")
    except Exception as e:
        logger.error(f"Failed to initialize transcriber: {e}")
        raise

def transcribe_audio(audio_data):
    if transcriber is None:
        raise RuntimeError("Transcriber not initialized")
    return transcriber.transcribe(audio_data)

if __name__ == "__main__":
    logger.info("Testing WhisperTranscriber initialization...")
    initialize_transcriber(model_size="tiny")
    logger.info("Initialization test complete.")
    
    test_audio_path = "path/to/your/test/audio.mp3"
    if os.path.exists(test_audio_path):
        with open(test_audio_path, "rb") as f:
            audio_data = f.read()
        transcription, language, confidence, segments = transcribe_audio(audio_data)
        logger.info(f"Transcription: {transcription}")
        logger.info(f"Language: {language} (confidence: {confidence})")
        logger.info("Segments:")
        for segment in segments:
            logger.info(f"[{segment.start:.2f}s -> {segment.end:.2f}s] {segment.text}")
    else:
        logger.warning(f"Test audio file not found: {test_audio_path}")