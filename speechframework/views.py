from django.http import HttpResponse
from django.template import loader
import pyaudio
import torch
from speechframework.models import SpeechClassifierModel, ConformerModel
import numpy as np
import wave
from speechbrain.processing.features import STFT, spectral_magnitude, Filterbank
import torchaudio
from torch.nn.utils.rnn import pack_sequence
from transformers import BertTokenizer
from django.shortcuts import render
import torch
import torch.nn as nn
TOK = BertTokenizer.from_pretrained('bert-base-uncased')
from whisperX import whisperx

model_asr = whisperx.load_model("small", 'cpu', compute_type="int8", language="en")
whisperx_align, metadata = whisperx.load_align_model(language_code="en", device="cpu")
model_vad, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                              model='silero_vad',
                              force_reload=True)

FORMAT = pyaudio.paInt16
CHANNELS = 1
SAMPLE_RATE = 16000
CHUNK = 1024

def home(request):
    template = loader.get_template('main.html')
    return HttpResponse(template.render())

def save_file(data):
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

def audio_main(request):
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
    newpage = work_method(stream, model, request)
    if newpage:
        return render(request, template_name='page.html')
   
def work_method(stream, model, request):
    frames = []
    data = []
    res = True
    while res:
        curr_chunk = stream.read(1024, exception_on_overflow = False)
        data.append(curr_chunk)
        print("....waiting....", end='')
        wake_word_detector = process_frame(curr_chunk,frames, data, 0, 0, model, request)
        if wake_word_detector:
            res = False
            wav_file = save_file(data)
            audio = whisperx.load_audio(wav_file)
            output = model_asr.transcribe(audio, 2)
            text = request.body.decode()
            print(output)
            print(text)
            if output['segments'] == text:
                print("The wake word has been detected")
                return wake_word_detector
            if output['segments'] != text:
                print("******************************", end='')

def process_frame(indata, frames,data, time, status, model, request):
    indata = np.frombuffer(indata, np.int8)
    a = int(indata.size)
    if a < 1024:
        indata = np.pad(indata, (0, 1024-a), mode='constant', constant_values=0)    
    frames.append(indata)  # Apply windowing function to the input frame
    num_frames = 16
    if len(frames) >= num_frames:
        wav_file = save_file(data)
        data.pop(0)
        frames.pop(0)  # Clear the frames list
        filterbank = Filterbank(n_mels=40)
        stft = STFT(sample_rate=16000, win_length=25, hop_length=10, n_fft=400)
        wavform, sr = torchaudio.load(wav_file)
        wavform = wavform.type('torch.FloatTensor')
        if sr > 16000:
            wavform = torchaudio.transforms.Resample(sr, 16000)(wavform)
        features = stft(wavform)
        features = spectral_magnitude(features)
        features = filterbank(features)
        features = pack_sequence(features, enforce_sorted=False)
        text = request.body.decode()
        text = TOK(text).input_ids
        text = torch.tensor(text)
        text = [text]
        text = pack_sequence(text, enforce_sorted=False)
        with torch.no_grad():
            output = model(features, text)
        
        best = np.where(output < 0.75, 0, 1)
        if best !=1:
            print(".................")   
        else:
            print(output)
            print("Detected", end='')
            return True
        
def int2float(sound):
    abs_max = np.abs(sound).max()
    sound = sound.astype('float32')
    if abs_max > 0:
        sound *= 1/32768
    sound = sound.squeeze()  # depends on the use case
    return sound

(get_speech_timestamps,
 save_audio,
 read_audio,
 VADIterator,
 collect_chunks) = utils

vad_iterator = VADIterator(model_vad)


def page_method(request):
    wake_word_detector = True if request.method == 'POST' else False
    audio = pyaudio.PyAudio()
    stream = audio.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=SAMPLE_RATE,
                    input=True,
                    frames_per_buffer=1024)
    continue_recording = True if wake_word_detector else False
    if wake_word_detector:
        print("Connected to the server")
        data = []
        count = 0
    while continue_recording:
        audio_chunk = stream.read(1024, exception_on_overflow = False)
        data.append(audio_chunk)
        if len(audio_chunk) == 0:
            break
        audio_int16 = np.frombuffer(audio_chunk, np.int16)
        audio_float32 = int2float(audio_int16)
        temp = torch.from_numpy(audio_float32)
        a = int(temp.size()[0])
        if a < 512: 
            m = nn.ConstantPad1d((0,512-a), 0.0000000)
            temp = m(temp)
        try:
            new_confidence =  model_vad(temp, 16000).item()
        except:
            save_file()
        value = round(new_confidence)
        if value == 0:
            count += 1
            print('.', sep=' ', end='', flush=True)
        else:
            count = 0  # Reset the count if the value is not 0
        if count == 80:
            print(str(count) + " milliseconds elapsed during waiting")
            res = save_file(data)
            continue_recording = False
            audio = whisperx.load_audio(res)
            result = model_asr.transcribe(audio, 2)
            result2 = whisperx.align(result["segments"], whisperx_align, metadata, audio, 'cpu', return_char_alignments=False)
            print(result["segments"][0]["text"]) if result["segments"] else print("No speech detected")
            print(result2["word_segments"])
            continue_recording = False
            stream.stop_stream()
            stream.close()
            #audio.terminate()
            results = result2["word_segments"]
            print(results)
            return render(request, 'page.html', context={'results': results})
    
