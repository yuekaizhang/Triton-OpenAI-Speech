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
    "voice": "default_zh",
    "response_format": "wav"
    }' \
    --output output.wav

curl $OPENAI_API_BASE/audio/speech \
    -H "Content-Type: application/json" \
    -d '{
    "model": "spark_tts",
    "input": "身临其境，换新体验。塑造开源语音合成新范式，让智能语音更自然。",
    "voice": "wukong",
    "response_format": "wav"
    }' \
    --output output2.wav

curl $OPENAI_API_BASE/audio/speech \
    -H "Content-Type: application/json" \
    -d '{
    "model": "spark_tts",
    "input": "身临其境，换新体验。塑造开源语音合成新范式，让智能语音更自然。",
    "voice": "leijun",
    "response_format": "wav"
    }' \
    --output output3.wav

# output3 from pcm
curl $OPENAI_API_BASE/audio/speech \
    -H "Content-Type: application/json" \
    -d '{
    "model": "spark_tts",
    "input": "身临其境，换新体验。塑造开源语音合成新范式，让智能语音更自然。",
    "voice": "leijun",
    "response_format": "pcm"
    }' | \
sox -t raw -r 16000 -e signed-integer -b 16 -c 1 - output3_from_pcm.wav

# load input from long_input.txt
input=$(cat long_input.txt)
# Construct JSON payload using jq
json_payload=$(jq -n --arg input_text "$input" '{model: "spark_tts", input: $input_text, voice: "default_zh", response_format: "wav"}')

curl $OPENAI_API_BASE/audio/speech \
    -H "Content-Type: application/json" \
    -d "$json_payload" \
    --output output4.wav

json_payload=$(jq -n --arg input_text "$input" '{model: "spark_tts", input: $input_text, voice: "default_zh", response_format: "pcm"}')
curl $OPENAI_API_BASE/audio/speech \
    -H "Content-Type: application/json" \
    -d "$json_payload" | \
sox -t raw -r 16000 -e signed-integer -b 16 -c 1 - output4_from_pcm.wav