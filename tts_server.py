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
from tts_frontend import TextNormalizer
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
    response_format: Optional[str] = "pcm" # Added: default to raw pcm stream, allow "wav"

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
text_normalizer = TextNormalizer()

async def _stream_audio_generator(request_data: TTSRequest):
    """Async generator to yield audio numpy arrays (int16) for each sentence."""
    # Initial checks (voice, ref audio) are now done in the main endpoint.

    config = VOICE_CONFIG[request_data.voice]
    reference_audio_path = config["reference_audio"]
    reference_text = config["reference_text"]
    request_model_name = request_data.model
    triton_url = f"{TRITON_SERVER_URL}/v2/models/{request_model_name}/infer"
    target_text_list = text_normalizer.text_normalize(request_data.input)

    try:
        # Read reference audio once
        waveform, sr = sf.read(reference_audio_path)
        # Sample rate check already done in the main endpoint

        # Ensure reference samples are float32 as expected by prepare_tts_request
        samples = np.array(waveform, dtype=np.float32)

        for target_text in target_text_list:
            print(f"Generating audio array for: {target_text}") # Log start of processing
            triton_request_data = prepare_tts_request(samples, reference_text, target_text, DEFAULT_SAMPLE_RATE)

            try:
                rsp = requests.post(
                    triton_url,
                    headers={"Content-Type": "application/json"},
                    json=triton_request_data,
                    # Consider adding a timeout
                    # timeout=30 # Example: 30 seconds
                )
                rsp.raise_for_status() # Raise exception for bad status codes (4xx or 5xx)
                result = rsp.json()

                if "error" in result:
                    print(f"Triton server error for text '{target_text}': {result['error']}")
                    continue # Skip yielding audio for this failed sentence

                if not result.get("outputs") or not result["outputs"][0].get("data"):
                    print(f"Invalid response structure from Triton for text '{target_text}'")
                    continue # Skip yielding audio for this invalid response

                audio_data = result["outputs"][0]["data"]
                # Assuming Triton returns float32 data
                audio_array = np.array(audio_data, dtype=np.float32)

                # Convert to 16-bit PCM
                audio_array = np.clip(audio_array, -1.0, 1.0)
                pcm_data = (audio_array * 32767).astype(np.int16)

                yield pcm_data # Yield the numpy array directly

            except requests.exceptions.Timeout:
                print(f"Triton request timed out for text '{target_text[:50]}...'")
                raise HTTPException(status_code=504, detail="Triton server request timed out during streaming.")
            except requests.exceptions.RequestException as e:
                print(f"Could not connect to Triton server for text '{target_text[:50]}...': {e}")
                raise HTTPException(status_code=503, detail=f"Could not connect to Triton server during streaming: {e}")
            except Exception as e:
                print(f"An unexpected error occurred processing text '{target_text[:50]}...': {str(e)}")
                raise HTTPException(status_code=502, detail=f"An unexpected error occurred during streaming: {str(e)}")

        print("Finished generating all sentence arrays.")

    except sf.SoundFileError as e:
         print(f"Error reading reference audio within generator: {e}")
         raise HTTPException(status_code=501, detail=f"Could not read reference audio file during streaming: {reference_audio_path}")
    except Exception as e:
        print(f"Unexpected error at start of/during generator execution: {e}")
        raise HTTPException(status_code=500, detail=f"Unexpected generator error: {str(e)}")


@app.post("/audio/speech")
async def generate_speech(request_data: TTSRequest):
    # --- Perform initial checks --- (same as before)
    if request_data.voice not in VOICE_CONFIG:
        raise HTTPException(status_code=400, detail=f"Voice '{request_data.voice}' not found.")

    config = VOICE_CONFIG[request_data.voice]
    reference_audio_path = config["reference_audio"]

    try:
        if not os.path.exists(reference_audio_path):
             raise FileNotFoundError
        info = sf.info(reference_audio_path)
        if info.samplerate != DEFAULT_SAMPLE_RATE:
             raise HTTPException(status_code=500, detail=f"Reference audio sample rate ({info.samplerate}) does not match expected ({DEFAULT_SAMPLE_RATE}). Resampling not implemented yet.")
    except FileNotFoundError:
         raise HTTPException(status_code=501, detail=f"Reference audio file not found: {reference_audio_path}")
    except sf.SoundFileError:
         raise HTTPException(status_code=501, detail=f"Could not read reference audio file info (invalid format or corrupt?): {reference_audio_path}")
    except Exception as e:
         raise HTTPException(status_code=500, detail=f"Error checking reference audio file: {str(e)}")

    # --- Handle response format --- 
    if request_data.response_format:
        response_format = request_data.response_format.lower()
        if response_format not in ["pcm", "wav"]:
            response_format = "wav"
    else:
        response_format = "wav"

    if response_format == "pcm":
        print("Streaming raw PCM audio.")
        media_type = f"audio/L16;rate={DEFAULT_SAMPLE_RATE};channels=1"
        
        async def pcm_byte_stream_generator():
            """Consumes numpy arrays from the main generator and yields bytes."""
            async for pcm_array in _stream_audio_generator(request_data):
                 yield pcm_array.tobytes()
            print("Finished streaming PCM bytes.")
        
        return StreamingResponse(pcm_byte_stream_generator(), media_type=media_type)

    elif response_format == "wav":
        print("Generating buffered WAV file.")
        all_audio_arrays = []
        try:
            async for pcm_array in _stream_audio_generator(request_data):
                all_audio_arrays.append(pcm_array)
        except HTTPException as e:
             # If the generator itself raises an HTTP exception, re-raise it
             raise e
        except Exception as e:
            # Catch unexpected errors during array collection
            print(f"Unexpected error collecting audio arrays for WAV: {e}")
            raise HTTPException(status_code=500, detail=f"Error generating full WAV file: {str(e)}")
        
        if not all_audio_arrays:
            print("No audio data generated, returning empty WAV.")
            # Return an empty WAV or perhaps an error?
            # Let's return a minimal valid empty WAV
            wav_buffer = io.BytesIO()
            sf.write(wav_buffer, np.array([], dtype=np.int16), DEFAULT_SAMPLE_RATE, format='WAV', subtype='PCM_16')
            wav_buffer.seek(0)
            return StreamingResponse(wav_buffer, media_type="audio/wav")
            # Alternatively: raise HTTPException(status_code=500, detail="Failed to generate any audio data")

        try:
            final_audio_array = np.concatenate(all_audio_arrays)
            print(f"Concatenated audio array shape: {final_audio_array.shape}")
            
            wav_buffer = io.BytesIO()
            sf.write(wav_buffer, final_audio_array, DEFAULT_SAMPLE_RATE, format='WAV', subtype='PCM_16')
            wav_buffer.seek(0)
            
            print("Returning complete WAV file.")
            return StreamingResponse(wav_buffer, media_type="audio/wav")
        except Exception as e:
            print(f"Error concatenating or writing WAV file: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to create final WAV file: {str(e)}")

    else:
        raise HTTPException(status_code=400, detail=f"Unsupported response_format: '{request_data.response_format}'. Supported formats: 'pcm', 'wav'.")


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