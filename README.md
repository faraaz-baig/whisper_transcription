# Whisper Transcription API

This project implements a high-performance audio transcription API using Rust, Python, and the Whisper AI model. It combines the efficiency of Rust for the web server with the power of Python's machine learning ecosystem.

## Components

1. Rust Web Server (`src/main.rs`)
2. Python Whisper Integration (`python/whisper_ffi.py`)
3. Benchmark Script (`benchmark_transcription.py`)

## Prerequisites

- Rust (latest stable version)
- Python 3.8+
- uv (Python package manager)

## Setup

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/whisper-transcription-api.git
   cd whisper-transcription-api
   ```

2. Install Rust dependencies:
   ```
   cargo build
   ```

3. Set up the Python environment:
   ```
   uv venv
   source .venv/bin/activate  # On Windows use: .venv\Scripts\activate
   uv pip install -r requirements.txt
   ```

## Running the Server

1. Start the Rust server:
   ```
   cargo run
   ```

The server will start on `http://localhost:3000`.

## API Usage

Send a POST request to `http://localhost:3000/transcribe` with the following JSON body:

```json
{
  "audio_data": [<raw audio bytes as a list of integers>]
}
```

The API will respond with:

```json
{
  "transcription": "Transcribed text",
  "language": "Detected language",
  "confidence": 0.98
}
```

## Benchmarking

To benchmark the API performance:

1. Update the `audio_file_path` in `benchmark_transcription.py` to point to your test audio file.
2. Run the benchmark script:
   ```
   python benchmark_transcription.py
   ```

This will output performance statistics for both sequential and concurrent requests.

## Project Structure

```
whisper-transcription-api/
├── src/
│   └── main.rs
├── python/
│   └── whisper_ffi.py
├── Cargo.toml
├── pyproject.toml
├── requirements.txt
├── benchmark_transcription.py
└── README.md
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
