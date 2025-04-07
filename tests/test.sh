
# OPENAI_API_KEY=sk-
# OPENAI_API_BASE="https://aihubmix.com/v1"
# curl $OPENAI_API_BASE/audio/speech \
#     -H "Content-Type: application/json" \
#     -d '{
#     "model": "tts-1",
#     "input": "身临其境，换新体验。塑造开源语音合成新范式，让智能语音更自然。",
#     "voice": "coral"
#     }' \
#     --output output.wav

OPENAI_API_BASE="http://localhost:8080"

curl $OPENAI_API_BASE/audio/speech \
    -H "Content-Type: application/json" \
    -d '{
    "model": "spark_tts",
    "input": "身临其境，换新体验。塑造开源语音合成新范式，让智能语音更自然。",
    "voice": "default_zh"
    }' \
    --output output.wav