# Triton-OpenAI-Speech
OpenAI-Compatible Frontend for Triton Inference ASR/TTS Server

### Quick Start
Before starting, launch one of the supported ASR/TTS services using Docker Compose.
| Model Repo | Supported |
| --- |  -- |
| [Spark-TTS](https://github.com/SparkAudio/Spark-TTS/tree/main/runtime/triton_trtllm) | Yes |
|[F5-TTS](https://github.com/SWivid/F5-TTS/tree/main/src/f5_tts/runtime/triton_trtllm)| Yes |
|[Cosyvoice2](https://github.com/FunAudioLLM/CosyVoice/tree/main/runtime/triton_trtllm)| Yes |

Then, launch the OpenAI-compatible API bridge server.
```sh
docker compose up
```

### Simple Test
```sh
bash tests/test.sh
```
### Usage

```
tts_server.py [-h] [--host HOST] [--port PORT] [--url URL]
                     [--ref_audios_dir REF_AUDIOS_DIR]
                     [--default_sample_rate DEFAULT_SAMPLE_RATE]

options:
  -h, --help            show this help message and exit
  --host HOST           Host to bind the server to
  --port PORT           Port to bind the server to
  --url URL             Triton server URL
  --ref_audios_dir REF_AUDIOS_DIR
                        Path to reference audio files
  --default_sample_rate DEFAULT_SAMPLE_RATE
                        Default sample rate
```