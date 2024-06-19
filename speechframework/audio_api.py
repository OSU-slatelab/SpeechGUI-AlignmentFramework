import whisperx
import pyaudio
import torch
from speechframework.models import ConformerModel
import numpy as np
import wave
from speechbrain.processing.features import STFT, spectral_magnitude, Filterbank
import torchaudio
from torch.nn.utils.rnn import pack_sequence
from transformers import BertTokenizer
import torch.nn as nn
import sys
from string import punctuation
from pdb import set_trace as bp


TOK = BertTokenizer.from_pretrained('bert-base-uncased')
model_asr = whisperx.load_model("small", 'cpu', compute_type="int8", language="en")
whisperx_align, metadata = whisperx.load_align_model(language_code="en", device="cpu")
model_vad, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                                model='silero_vad',
                                force_reload=True)
FORMAT = pyaudio.paInt16
CHANNELS = 1
SAMPLE_RATE = 16000
CHUNK = 1024
(get_speech_timestamps,
 save_audio,
 read_audio,
 VADIterator,
 collect_chunks) = utils

vad_iterator = VADIterator(model_vad)


def save_file(data):
    """
    Saves the audio file to the disk
    :param data: List of audio data chunks
    :return: The name of the saved file
    
    """
    output_file = "my_voice.wav"
    sample_width = 2  # 2 bytes for 16-bit audio, 1 byte for 8-bit audio
    with wave.open(output_file, 'wb') as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(sample_width)
        wf.setframerate(SAMPLE_RATE)

        for chunk in data:
            wf.writeframes(chunk)
        wf.close()
    return output_file

def load_wavform(file):
    """
    Loads wavform file with a sample rate of 16000
    :file: The path to the file to load
    :return: The loaded wavform object
    """
    wavform, sr = torchaudio.load(file)

    wavform = wavform.type('torch.FloatTensor')
    if sr > 16000:
        wavform = torchaudio.transforms.Resample(sr, 16000)(wavform)
    return wavform


def get_features(wavform):
    """
    Extracts features from the audio data
    :param wavform: The audio data
    :return: The extracted features
    """
    filterbank = Filterbank(n_mels=40)
    stft = STFT(sample_rate=16000, win_length=25, hop_length=10, n_fft=400)

    features = stft(wavform)
    features = spectral_magnitude(features)
    features = filterbank(features)
    features = pack_sequence(features, enforce_sorted=False)

    return features

def get_text(request):
    """
    Tokenizes the text
    :param request: The request object
    :return: The tokenized text
    
    """
    text = request.body.decode() 
    text = TOK(text).input_ids
    text = torch.tensor(text)
    text = [text]
    text = pack_sequence(text, enforce_sorted=False)
    return text


def get_match_probs(indata, frames, data, model, request):
    """
    If enough frames are collected, gets the probability they match the wake word
    :param indata: The current input chunk of audio data
    :param frames: The list of frames for collecting the audio data
    :param data: The audio data accumulated so far
    :param model: The wake word detection model
    :param request: The request object
    :return: The probability the wake word is present in the given audio segments or None if not enough frames
    """
    # get most recent audio chunk into np array of size 1024
    indata = np.frombuffer(indata, np.int8)
    a = int(indata.size)
    if a < 1024:
        indata = np.pad(indata, (0, 1024-a), mode='constant', constant_values=0)

    # add the chunk to our frames
    frames.append(indata)  # Apply windowing function to the input frame
    num_frames = 16

    # if we have enough frames, we can run the model
    output = None
    if len(frames) >= num_frames:
        # save the audio data to a file
        wav_file = save_file(data)
        data.pop(0)
        # clear the frames list
        frames.pop(0)
        wavform = load_wavform(wav_file)
        features = get_features(wavform)
        # The word selected by the user on the front end
        text = get_text(request)
        with torch.no_grad():
            output = model(features, text)
    return output
    

def check_for_word(indata, frames, data, model, request):
    """
    Determines if a word was found
    :param indata: The current input chunk of audio data
    :param frames: The list of frames for collecting the audio data
    :param data: The audio data accumulated so far
    :param model: The wake word detection model
    :param request: The request object
    :returns: whether a word was found and the match probability if it was found (otherwise None)
    """
    output = get_match_probs(indata, frames, data, model, request)
    # If the model outputs the word is detected with a confidence of 0.75 or more, we return True
    if output is None or np.where(output < 0.75, 0, 1) != 1:
        return False, None
    else:
        print(f"\nDetected a word with {output} confidence")
        return True, output


def strip_format(s):
    s = s.translate(str.maketrans('', '', punctuation))
    s = s.strip()
    s = s.lower()
    return s

def wait_to_wake(stream, model, request):
    """
    Waits for the wake word 
    :param stream: The open stream object
    :param model: Wake word detection model
    :param request: The request object
    :return: The presence/absence of the wake word after double checking the transcription and the associated probabilities
    :description: This method processes the incoming audio frames to look for the wake word
    """
    frames = []
    data = []
    res = True
    print("Now waiting....")
    while res:
        # add the current chunk to the data list
        curr_chunk = stream.read(1024, exception_on_overflow=False)
        data.append(curr_chunk)
        # check if the wake word is detected
        is_word, wake_probs = check_for_word(curr_chunk, frames, data, model, request)
        if is_word:
            # save the audio data to a file
            wav_file = save_file(data)
            audio = whisperx.load_audio(wav_file)
            # transcribe the audio
            output = model_asr.transcribe(audio, 2)
            output = output["segments"]
            print(f"Detected Word: {strip_format(output[0]['text']) if output != [] else '[Whisper found no speech]'}")
            text = request.body.decode()
            text = text[text.find(':') + 2 :]
            text = text[: text.find('"')]
            print(f"Wake Work: {strip_format(text)}")
            if output != [] and strip_format(output[0]["text"]) == strip_format(text):
                res = False
                print("The wake word has been detected")
                return True, wake_probs
            else:
                print("******************************")
                # return False, wake_probs


def start_wake_detection(request):
    """
    Initialize audio processes in IKLE mode and wait for wake word
    :param request: The request object
    :return: None
    :description: This method opens the audio stream, loads the wake word model and waits for the wake word to return

    """
    audio = pyaudio.PyAudio()
    stream = audio.open(format=FORMAT,
                        channels=CHANNELS,
                        rate=SAMPLE_RATE,
                        input=True,
                        frames_per_buffer=1024)
    device = torch.device('cpu')
    checkpoint_path = "word_level_train_015_256_bert1_st00_2fc_conf_16_sig_bce_rop_80.pt"
    pretrained_model = torch.load(checkpoint_path, map_location=device)
    model_state_dict = pretrained_model["model_state_dict"]
    model_params = {'num_classes': pretrained_model['model_params']['num_classes'],
                    'feature_size': pretrained_model['model_params']['feature_size'],
                    'hidden_size': pretrained_model['model_params']['hidden_size'],
                    'num_layers': pretrained_model['model_params']['num_layers'],
                    'dropout': pretrained_model['model_params']['dropout'],
                    'bidirectional': pretrained_model['model_params']['bidirectional'],
                    'device': device}
    model = ConformerModel(**model_params)
    model.load_state_dict(model_state_dict)
    model.eval()
    print("Waiting for the wake word to start the recording (Wake word detector):")
    is_wake_word, wake_probs = wait_to_wake(stream, model, request)
    return wake_probs


def int2float(sound):
    """
    Convert the integer audio data to float
    :param sound: The audio data
    :return: The audio data in float format
    """
    abs_max = np.abs(sound).max()
    sound = sound.astype('float32')
    if abs_max > 0:
        sound *= 1/32768
    sound = sound.squeeze()  # depends on the use case
    return sound


def get_latest_audio(stream, data):
    """
    Grabs the latest audio data
    :param stream: The open stream object
    :param data: The audio data accumulated so far
    :return: The latest audio data
    """
    audio_chunk = stream.read(1024, exception_on_overflow=False)
    data.append(audio_chunk)
    audio_int16 = np.frombuffer(audio_chunk, np.int16)
    audio_float32 = int2float(audio_int16)
    audio_float32 = torch.from_numpy(audio_float32)
    dim1 = int(audio_float32.size()[0])
    if dim1 < 512:
        m = nn.ConstantPad1d((0, 512-dim1), 0.0000000)
        audio_float32 = m(audio_float32)
    return audio_float32


def transcribe_audio(request):
    """
    Records and transcribes the audio
    Should be called after the wake word is detected
    :param request: The request object
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
    
    if request.method == 'POST':
        is_wake_word = True
        data = []
        count = 0
        continue_recording = True 
    else:
        is_wake_word = False
        continue_recording = False

    while continue_recording:
        # Record the incoming audio
        audio_float32 = get_latest_audio(stream, data)
        
        # Check if the audio is silent and if so count the number of milliseconds
        try:
            new_confidence = model_vad(audio_float32, 16000).item()
        except Exception:
            save_file()
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
            print(results)
            return results


def wake_and_asr(request):
    """
    Wait for wake word and then transcribe the audio
    :param request: The request object
    :return: The transcription of the audio
    """
    start_wake_detection(request)
    results = transcribe_audio(request)
    return results