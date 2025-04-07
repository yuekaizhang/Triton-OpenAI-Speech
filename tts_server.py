import requests
import soundfile as sf
import json
import numpy as np
import argparse
import io
import os  # Added import
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import uvicorn
from pathlib import Path
from typing import Optional # Added import

def register_voice(ref_audios_path):
    VOICE_CONFIG = {}
    for ref_audio in os.listdir(ref_audios_path):
        if ref_audio.endswith(".wav") or ref_audio.endswith(".mp3"):
            voice_name = Path(ref_audio).stem
            reference_text = open(os.path.join(ref_audios_path, ref_audio.replace(".wav", ".txt")), "r").read()
            VOICE_CONFIG[voice_name] = {
                "reference_audio": os.path.join(ref_audios_path, ref_audio),
                "reference_text": reference_text
            }
    return VOICE_CONFIG

class TTSRequest(BaseModel):
    model: str # We might not use this directly if mapping voice to model config
    input: str
    voice: str
    instructions: Optional[str] = None # Optional field if needed

def prepare_tts_request(
    waveform,
    reference_text,
    target_text,
    sample_rate,
):
    assert len(waveform.shape) == 1, "waveform should be 1D"
    lengths = np.array([[len(waveform)]], dtype=np.int32)

    samples = waveform.reshape(1, -1).astype(np.float32)

    data = {
        "inputs":[
            {
                "name": "reference_wav",
                "shape": samples.shape,
                "datatype": "FP32",
                "data": samples.tolist()
            },
            {
                "name": "reference_wav_len",
                "shape": lengths.shape,
                "datatype": "INT32",
                "data": lengths.tolist(),
            },
            {
                "name": "reference_text",
                "shape": [1, 1],
                "datatype": "BYTES",
                "data": [reference_text]
            },
            {
                "name": "target_text",
                "shape": [1, 1],
                "datatype": "BYTES",
                "data": [target_text]
            }
        ]
    }
    return data


app = FastAPI()

@app.post("/audio/speech")
async def generate_speech(request_data: TTSRequest):
    if request_data.voice not in VOICE_CONFIG:
        raise HTTPException(status_code=400, detail=f"Voice '{request_data.voice}' not found.")
    config = VOICE_CONFIG[request_data.voice]
    reference_audio_path = config["reference_audio"]
    reference_text = config["reference_text"]
    target_text = request_data.input

    try:

        waveform, sr = sf.read(reference_audio_path)

        if sr != DEFAULT_SAMPLE_RATE:
             raise HTTPException(status_code=500, detail=f"Reference audio sample rate ({sr}) does not match expected ({DEFAULT_SAMPLE_RATE}). Resampling not implemented yet.")


        samples = np.array(waveform, dtype=np.float32)

        triton_request_data = prepare_tts_request(samples, reference_text, target_text, DEFAULT_SAMPLE_RATE)
        request_model_name = request_data.model
        triton_url = f"{TRITON_SERVER_URL}/v2/models/{request_model_name}/infer"

        rsp = requests.post(
            triton_url,
            headers={"Content-Type": "application/json"},
            json=triton_request_data,
        )
        rsp.raise_for_status() # Raise exception for bad status codes (4xx or 5xx)
        result = rsp.json()

        if "error" in result:
             raise HTTPException(status_code=500, detail=f"Triton server error: {result['error']}")

        if not result.get("outputs") or not result["outputs"][0].get("data"):
             raise HTTPException(status_code=500, detail="Invalid response structure from Triton server.")

        audio_data = result["outputs"][0]["data"]
        audio_array = np.array(audio_data, dtype=np.float32)
        audio_buffer = io.BytesIO()
        sf.write(audio_buffer, audio_array, DEFAULT_SAMPLE_RATE, format='WAV', subtype='PCM_16')
        audio_buffer.seek(0)

        return StreamingResponse(audio_buffer, media_type="audio/wav")

    except sf.SoundFileError:
        raise HTTPException(status_code=501, detail=f"Could not read reference audio file: {reference_audio_path}")
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=503, detail=f"Could not connect to Triton server: {e}")
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"An unexpected error occurred: {str(e)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser() 
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind the server to")
    parser.add_argument("--port", type=int, default=8080, help="Port to bind the server to")
    parser.add_argument("--url", type=str, default="http://localhost:8000", help="Triton server URL")
    parser.add_argument("--ref_audios_dir", type=str, default="./ref_audios", help="Path to reference audio files")
    parser.add_argument("--default_sample_rate", type=int, default=16000, help="Default sample rate")
    args = parser.parse_args()

    VOICE_CONFIG = register_voice(args.ref_audios_dir)
    TRITON_SERVER_URL = args.url
    DEFAULT_SAMPLE_RATE = args.default_sample_rate
    REF_AUDIO_BASE_PATH = args.ref_audios_dir
    args = parser.parse_args()

    print(f"Starting FastAPI server on {args.host}:{args.port}")
    print(f"Using Triton server at {TRITON_SERVER_URL}")
    print(f"Available voices: {list(VOICE_CONFIG.keys())}")

    uvicorn.run(app, host=args.host, port=args.port) 