import cv2
import websockets 
import asyncio
import base64
import json
import sys
import numpy as np

from gesture_api import start_gesture_detection, transcribe_audio_gesture

def main():
    gesture = sys.argv[1]
    """
    Wait for wake gesture and then transcribe the audio
    :return: The transcription of the audio
    """
    results = start_gesture_detection(gesture=gesture, url='ws://localhost:5000')
    # results = transcribe_audio_gesture()
    return results

if __name__ == "__main__":
    while True:
        main()