import cv2
import websockets 
import asyncio
import base64
import json
import pyaudio
import whisperx
import torch
import numpy as np

from speechframework.audio_api import get_latest_audio, save_file

FORMAT = pyaudio.paInt16
CHANNELS = 1
SAMPLE_RATE = 16000
model_asr = whisperx.load_model("small", 'cpu', compute_type="int8", language="en")
whisperx_align, metadata = whisperx.load_align_model(language_code="en", device="cpu")
model_vad, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                                model='silero_vad',
                                force_reload=True)

async def send_video(gesture, url):
    while True:
        try: 
            # Use websockets to connect to URL and port on server
            async with websockets.connect(url) as websocket:
                # Opens camera for video capture
                cap = cv2.VideoCapture(0)
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break

                    # Encode frame as JPEG
                    _, buffer = cv2.imencode('.jpg', frame)
                    frame_data = base64.b64encode(buffer).decode('utf-8')

                    # Stores frame data into json to send to server
                    payload = json.dumps({'frame': frame_data})

                    # Sends frame to the server
                    await websocket.send(payload)

                    response = await websocket.recv()
                    result = json.loads(response)

                    output_frame_data = result['frame']
                    output_frame = np.frombuffer(base64.b64decode(output_frame_data), dtype=np.uint8)
                    output_frame = cv2.imdecode(output_frame, cv2.IMREAD_COLOR)

                    # cv2.imshow('frame', output_frame)
                    # cv2.waitKey(1) 
                    
                    result_gesture = result['gesture']
                    result_gesture_score = result['gesture_score']
                    result_engagement = result['engagement']
                    result_engagement_score = result['engagement_score']
                    # result_age = result['age']
                    # result_age_score = result['age_score']

                    # print(f"Gesture: {result_gesture}\nGesture Score: {result_gesture_score}\nEngagement: {result_engagement}\nEngagement Score: {result_engagement_score}\nAge: {result_age}\nAge Score: {result_age_score}")
                    print(f"Gesture: {result_gesture}\nGesture Score: {result_gesture_score}\nEngagement: {result_engagement}\nEngagement Score: {result_engagement_score}\n")

                    if gesture in result_gesture:
                        print('Wake gesture detected')
                        results = transcribe_audio_gesture()
                        return results 

                cap.release()
        except websockets.exceptions.ConnectionClosedError:
            print("Connection closed, retrying...")
            await asyncio.sleep(1)  # Wait before retrying
        except Exception as e:
            print(f"An error occurred: {e}")
            await asyncio.sleep(1)  # Wait before retrying

def start_gesture_detection(gesture, url):
    """
    Initialize processes and wait for wake gesture
    :param gesture: The chosen wake gesture
    :return: None
    :description: This method opens the websocket connection between client and server and waits for wake gesture output in a while loop
    """

    recognized_gestures = ["Closed_Fist", "Open_Palm", "Pointing_Up", "Thumb_Down", "Thumb_Up", "Victory", "ILoveYou"]

    assert gesture in recognized_gestures, "request must be one of the recognized Mediapipe gestures: Closed_Fist, Open_Palm, Pointing_Up, Thumb_Down, Thumb_Up, Victory, or ILoveYou"

    print("Waiting for the wake gesture to start the recording: ")
    current_frame_gesture = "Temp"
    current_frame_gesture_score = 0

    # add way to check confidence
    results = asyncio.get_event_loop().run_until_complete(send_video(gesture, url))
    return results 

def transcribe_audio_gesture():
    """
    Records and transcribes the audio
    Should be called after the wake word is detected
    :return: The transcription of the audio
    :description: Record the incoming audio, till 5 second of silence is detected, 
    then transcribe the audio and align the transcription with the audio
    """
    audio = pyaudio.PyAudio()
    stream = audio.open(format=FORMAT,
                        channels=CHANNELS,
                        rate=SAMPLE_RATE,
                        input=True,
                        frames_per_buffer=1024)
    
    data = []
    count = 0
    continue_recording = True

    while continue_recording:
        # Record the incoming audio
        audio_float32 = get_latest_audio(stream, data)

        # Check if the audio is silent and if so count the number of milliseconds
        try:
            new_confidence = model_vad(audio_float32, 16000).item()
            print(new_confidence)
        except Exception as e:
            print(f"An error occurred: {e}")
            save_file(data)
        is_voice = round(new_confidence)
        if is_voice == 0:
            count += 1
            print('.', sep=' ', end='', flush=True)
        else:
            count = 0  # Reset the count if the value is not 0

        # If 80 milliseconds of silence is detected, stop the recording
        if count == 80:
            print(str(count) + " milliseconds elapsed during waiting")
            res = save_file(data)
            continue_recording = False
            # Transcribe the audio and align the transcription with the audio
            audio = whisperx.load_audio(res)
            result = model_asr.transcribe(audio, 2)
            result2 = whisperx.align(result["segments"], whisperx_align, metadata, audio, 'cpu', return_char_alignments=False)
            print(result["segments"][0]["text"]) if result["segments"] else print("No speech detected")
            print(result2["word_segments"])
            continue_recording = False
            stream.stop_stream()
            stream.close()
            results = result2["word_segments"]
            return results
