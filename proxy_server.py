import os
import json
import asyncio
import websockets
import numpy as np
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("SONIOX_API_KEY")
PORT = int(os.getenv("PORT", 3001))
MODEL = "stt-rt-v3"  # Updated to the active v3 model for better multilingual support
DEBOUNCE_TIME = 0.5  # Seconds to wait before sending a transcription phrase
LANGUAGE = "ro"  # Romanian language code

async def process_soniox(ws, channel, vapi_ws):
    buffer = ""
    timer = None

    async def debounce():
        nonlocal buffer  # Declare at the top before any use
        await asyncio.sleep(DEBOUNCE_TIME)
        if buffer:
            await vapi_ws.send(json.dumps({
                "type": "transcriber-response",
                "transcription": buffer.strip(),
                "channel": channel
            }))
            buffer = ""

    async for message in ws:
        result = json.loads(message)
        added = False
        for token in result.get("tokens", []):
            if token.get("is_final"):
                buffer += token["text"]
                added = True
        if added:
            if timer:
                timer.cancel()
            timer = asyncio.create_task(debounce())

    if timer:
        timer.cancel()
    if buffer:
        await vapi_ws.send(json.dumps({
            "type": "transcriber-response",
            "transcription": buffer.strip(),
            "channel": channel
        }))

async def handle_client(websocket):
    if websocket.path != "/api/custom-transcriber":
        await websocket.close()
        return

    print("VAPI connected")

    try:
        start_msg = await websocket.recv()
        start_data = json.loads(start_msg)
        if start_data["type"] != "start":
            return

        encoding = start_data["encoding"]
        sample_rate = start_data["sampleRate"]
        channels = start_data["channels"]

        if encoding != "linear16" or channels != 2 or sample_rate != 16000:
            print("Unsupported audio format")
            await websocket.close()
            return

        customer_ws = await websockets.connect("wss://stt-rt.soniox.com/transcribe-websocket")
        assistant_ws = await websockets.connect("wss://stt-rt.soniox.com/transcribe-websocket")

        soniox_start = {
            "api_key": API_KEY,
            "model": MODEL,
            "audio_format": encoding,
            "sample_rate": sample_rate,
            "num_channels": 1,  # Mono for each channel
            "language_hints": [LANGUAGE]  # For Romanian
        }

        await customer_ws.send(json.dumps(soniox_start))
        await assistant_ws.send(json.dumps(soniox_start))

        customer_task = asyncio.create_task(process_soniox(customer_ws, "customer", websocket))
        assistant_task = asyncio.create_task(process_soniox(assistant_ws, "assistant", websocket))

        async for data in websocket:
            if isinstance(data, bytes):
                audio_array = np.frombuffer(data, dtype=np.int16)
                customer_audio = audio_array[0::2].tobytes()
                assistant_audio = audio_array[1::2].tobytes()
                await customer_ws.send(customer_audio)
                await assistant_ws.send(assistant_audio)

    except Exception as e:
        print(f"Error: {e}")
    finally:
        await customer_ws.close()
        await assistant_ws.close()
        customer_task.cancel()
        assistant_task.cancel()

async def main():
    async with websockets.serve(handle_client, "0.0.0.0", PORT):
        print(f"Listening on port {PORT}")
        await asyncio.Future()  # Run forever

if __name__ == "__main__":
    asyncio.run(main())
