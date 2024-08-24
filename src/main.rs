use axum::{
    extract::State,
    routing::post,
    Json, Router,
};
use pyo3::prelude::*;
use pyo3::types::{PyTuple, PyList, PyBytes};
use tokio::time::{timeout, Duration, Instant};
use serde::{Deserialize, Serialize};
use std::{env, path::PathBuf, sync::Arc};

#[derive(Serialize)]
struct Segment {
    start: f32,
    end: f32,
    text: String,
}


#[derive(Serialize)]
struct TranscriptionResponse {
    transcription: String,
    language: String,
    confidence: f32,
    segments: Vec<Segment>,
}

struct AppState {
    whisper_module: Arc<PyObject>,
}

#[derive(Deserialize)]
struct TranscriptionRequest {
    audio_data: Vec<u8>,
}

fn print_debug_info(py: Python) {
    println!("Python version: {}", py.version());
    // println!("Python prefix: {}", py.prefix());
    if let Ok(sys) = py.import("sys") {
        if let Ok(path) = sys.getattr("path") {
            println!("Python sys.path: {:?}", path);
        }
    }
    println!("Current working directory: {:?}", std::env::current_dir().unwrap());
    println!("PYTHONPATH: {:?}", std::env::var("PYTHONPATH"));
}

fn add_python_path() -> PyResult<()> {
    Python::with_gil(|py| {
        let sys = py.import("sys")?;
        let current_dir = env::current_dir()?;
        let python_path = current_dir.join("python");
        println!("Adding to Python path: {:?}", python_path);
        sys.getattr("path")?.call_method1("insert", (0, python_path.to_str().unwrap()))?;
        Ok(())
    })
}

fn get_venv_python_path() -> Option<PathBuf> {
    let current_dir = env::current_dir().ok()?;
    let venv_path = current_dir.join(".venv");
    
    #[cfg(windows)]
    let python_path = venv_path.join("Scripts").join("python.exe");
    
    #[cfg(not(windows))]
    let python_path = venv_path.join("bin").join("python");

    if python_path.exists() {
        Some(python_path)
    } else {
        None
    }
}

#[tokio::main]
async fn main() -> PyResult<()> {
    // Set the Python interpreter path to use our uv-created venv
    if let Some(venv_python) = get_venv_python_path() {
        println!("Using Python from uv-created venv: {:?}", venv_python);
        env::set_var("PYTHONEXECUTABLE", venv_python);
    } else {
        println!("Warning: uv-created virtual environment Python not found. Using system Python.");
    }

    add_python_path()?;

    let whisper_module = Python::with_gil(|py| -> PyResult<PyObject> {
        print_debug_info(py);
        
        println!("Attempting to import whisper_ffi...");
        let whisper_module = py.import("whisper_ffi")?;
        println!("whisper_ffi imported successfully.");
        
        println!("Initializing transcriber...");
        whisper_module.getattr("initialize_transcriber")?.call0()?;
        println!("Transcriber initialized successfully.");
        
        Ok(whisper_module.into())
    }).map_err(|e| {
        eprintln!("Python error: {}", e);
        eprintln!("Make sure you have installed the required Python packages.");
        eprintln!("Try running: pip install faster-whisper");
        e
    })?;

    let state = Arc::new(AppState {
        whisper_module: Arc::new(whisper_module),
    });

    let app = Router::new()
        .route("/transcribe", post(transcribe))
        .with_state(state);

    let listener = tokio::net::TcpListener::bind("127.0.0.1:3000").await.unwrap();
    println!("listening on {}", listener.local_addr().unwrap());
    axum::serve(listener, app).await.unwrap();

    Ok(())
}

async fn transcribe(
    State(state): State<Arc<AppState>>,
    Json(payload): Json<TranscriptionRequest>,
) -> Json<TranscriptionResponse> {
    println!("Received audio data of length: {}", payload.audio_data.len());
    let start_time = Instant::now();

    let result = timeout(Duration::from_secs(300), async {
        Python::with_gil(|py| -> PyResult<(String, String, f32, Vec<Segment>)> {
            let whisper_module = state.whisper_module.as_ref();
            let audio_data = PyBytes::new(py, &payload.audio_data);
            println!("Calling transcribe_audio function...");
            let result = whisper_module
                .getattr(py,"transcribe_audio")?
                .call1(py,(audio_data,))?;
            
            println!("Transcribe_audio function call completed");
            let tuple: &PyTuple = result.downcast(py)?;
            let transcription: String = tuple.get_item(0)?.extract()?;
            let language: String = tuple.get_item(1)?.extract()?;
            let confidence: f32 = tuple.get_item(2)?.extract()?;
            let segments_py: &PyList = tuple.get_item(3)?.downcast()?;
            
            let segments: Vec<Segment> = segments_py
                .iter()
                .map(|seg| -> PyResult<Segment> {
                    Ok(Segment {
                        start: seg.getattr("start")?.extract()?,
                        end: seg.getattr("end")?.extract()?,
                        text: seg.getattr("text")?.extract()?,
                    })
                })
                .collect::<PyResult<Vec<Segment>>>()?;

            Ok((transcription, language, confidence, segments))
        })
    }).await;

    let result = match result {
        Ok(r) => r.unwrap_or_else(|e| {
            eprintln!("Error during transcription: {}", e);
            (
                "Transcription failed".to_string(),
                "unknown".to_string(),
                0.0,
                vec![],
            )
        }),
        Err(_) => {
            eprintln!("Transcription timed out after 300 seconds");
            (
                "Transcription timed out".to_string(),
                "unknown".to_string(),
                0.0,
                vec![],
            )
        }
    };

    let elapsed = start_time.elapsed();
    println!("Transcription completed in {:?}", elapsed);

    Json(TranscriptionResponse {
        transcription: result.0,
        language: result.1,
        confidence: result.2,
        segments: result.3,
    })
}