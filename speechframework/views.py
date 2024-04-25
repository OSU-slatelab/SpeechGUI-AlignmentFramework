from django.http import HttpResponse, JsonResponse
from django.template import loader
import socket               # Import socket module
import pyaudio
import torch
from speechframework.models import SpeechClassifierModel, ConformerModel
import numpy as np
import wave
from speechbrain.processing.features import STFT, spectral_magnitude, Filterbank, Deltas, InputNormalization, ContextWindow
import torchaudio
from torch.nn.utils.rnn import pack_sequence, pad_sequence
from django.views.decorators.csrf import csrf_protect
from transformers import BertTokenizer
from django.shortcuts import render
import ast
import json

TOK = BertTokenizer.from_pretrained('bert-base-uncased')


def home(request):
    template = loader.get_template('main.html')
    return HttpResponse(template.render())

def home1(request, results):
    import pdb;pdb.set_trace()
    template = loader.get_template('main.html')
    context = {'results': results}  # Create a context dictionary with the results
    return HttpResponse(template.render(context, request))  # Pass the context to the template
# Create your views here.

AUD =[]
@csrf_protect
def process_audio(request):
    global AUD
    checkpoint_path = "/Users/beulah_karrolla/Desktop/project/best_models/word_level_train_015_256_bert1_st00_2fc_conf_16_sig_bce_rop_80.pt"
    pretrained_model = torch.load(checkpoint_path, map_location='cpu')
    model_state_dict = pretrained_model["model_state_dict"]
    model_params = {'num_classes': pretrained_model['model_params']['num_classes'],
                        'feature_size': pretrained_model['model_params']['feature_size'],
                        'hidden_size': pretrained_model['model_params']['hidden_size'],
                        'num_layers': pretrained_model['model_params']['num_layers'],
                        'dropout': pretrained_model['model_params']['dropout'],
                        'bidirectional': pretrained_model['model_params']['bidirectional'],
                        'device': 'cpu'}
    model = ConformerModel(**model_params)
    model.load_state_dict(model_state_dict)
    model.eval()
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    res1 = request.body.decode()
    
    t1 =json.loads(res1)
    res = t1['res']
    text = t1['kwd']
    SAMPLE_RATE = 16000
    CHUNK = 1024 #int(SAMPLE_RATE / 10) # Still has to differntiate between various chunk sizes
    num_samples = 1536
    audio = pyaudio.PyAudio()
    stream = audio.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=SAMPLE_RATE,
                    input=True,
                    frames_per_buffer=1024)
    while res:
        data = stream.read(1024, exception_on_overflow = False)
        AUD.append(data)
    if not res:
        stream.stop_stream()
        stream.close()
        audio.terminate()
        print(len(AUD))
        filename = save_file(AUD)
        AUD = []
        output = process_frame_new(filename, model, text)
        return render(request, template_name='main.html', context={'output': output})

def process_frame_new(file, model, text):
        filterbank = Filterbank(n_mels=40)
        stft = STFT(sample_rate=16000, win_length=25, hop_length=10, n_fft=400)
        wavform, sr = torchaudio.load(file)

        wavform = wavform.type('torch.FloatTensor')
        if sr > 16000:
            wavform = torchaudio.transforms.Resample(sr, 16000)(wavform)
        features = stft(wavform)
        features = spectral_magnitude(features)
        features = filterbank(features)
        features = pack_sequence(features, enforce_sorted=False)
        text = text
        #print(TOK.convert_ids_to_tokens(TOK(text).input_ids))
        text = TOK(text).input_ids
        text = torch.tensor(text)
        text = [text]
        text = pack_sequence(text, enforce_sorted=False)
        
        
        with torch.no_grad():
            output = model(features, text)
        return output
        
    
    

def save_file(data):
    output_file = "my_voice.wav"
    sample_width = 2  # 2 bytes for 16-bit audio, 1 byte for 8-bit audio
    CHANNELS = 1
    SAMPLE_RATE = 16000
    #CHUNK = int(SAMPLE_RATE / 10)
    with wave.open(output_file, 'wb') as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(sample_width)
        wf.setframerate(SAMPLE_RATE)

        for chunk in data:
            wf.writeframes(chunk)
        wf.close()
    return output_file

def process_frame(indata, frames,data, time, status, model, request):
    indata = np.frombuffer(indata, np.int8)
    a = int(indata.size)
    if a < 1024:
        indata = np.pad(indata, (0, 1024-a), mode='constant', constant_values=0)
        '''m = nn.ConstantPad1d((0,512-a), 0.0000000)
        indata = m(indata)'''
    frames.append(indata)  # Apply windowing function to the input frame
    # Check if enough frames are collected
    num_frames = 16
    if len(frames) >= num_frames:
        #import ipdb;ipdb.set_trace()
        wav_file = save_file(data)
        data.pop(0)
        #frames_array = np.array(frames[:num_frames])  # Convert frames list to numpy array
        frames.pop(0)  # Clear the frames list
        filterbank = Filterbank(n_mels=40)
        stft = STFT(sample_rate=16000, win_length=25, hop_length=10, n_fft=400)
        deltas = Deltas(input_size=40)
        #self.context_window = ContextWindow(window_size=151, left_frames=75, right_frames=75)
        input_norm = InputNormalization()
        wavform, sr = torchaudio.load(wav_file)
        #wavform = self.input_norm(wavform)
        wavform = wavform.type('torch.FloatTensor')
        if sr > 16000:
            wavform = torchaudio.transforms.Resample(sr, 16000)(wavform)
        features = stft(wavform)
        features = spectral_magnitude(features)
        features = filterbank(features)
        features = pack_sequence(features, enforce_sorted=False)
        text = request.body.decode()
        #print(TOK.convert_ids_to_tokens(TOK(text).input_ids))
        text = TOK(text).input_ids
        text = torch.tensor(text)
        text = [text]
        text = pack_sequence(text, enforce_sorted=False)
        
        
        with torch.no_grad():
            output = model(features, text)
        
        best = np.where(output < 0.999, 0, 1)
        
        temp2 = best
        if temp2 !=1:
            print(".................")
            
        if temp2 == 1:
            print(output)
            print("Detected", end='')
            return True
            #import ipdb;ipdb.set_trace()
            #audio = whisperx.load_audio(wav_file)
        
def audio_main(request):
    global continue_recording
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    SAMPLE_RATE = 16000
    CHUNK = 1024 #int(SAMPLE_RATE / 10) # Still has to differntiate between various chunk sizes
    num_samples = 1536

    audio = pyaudio.PyAudio()
    stream = audio.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=SAMPLE_RATE,
                    input=True,
                    frames_per_buffer=1024)

    #data = []

    frames = []
    device = torch.device('cpu')
    checkpoint_path = "/Users/beulah_karrolla/Desktop/project/best_models/word_level_train_015_256_bert1_st00_2fc_conf_16_sig_bce_rop_80.pt"
    pretrained_model = torch.load(checkpoint_path, map_location=device)
    model_state_dict = pretrained_model["model_state_dict"]
    model_params = {'num_classes': pretrained_model['model_params']['num_classes'],
                        'feature_size': pretrained_model['model_params']['feature_size'],
                        'hidden_size': pretrained_model['model_params']['hidden_size'],
                        'num_layers': pretrained_model['model_params']['num_layers'],
                        'dropout': pretrained_model['model_params']['dropout'],
                        'bidirectional': pretrained_model['model_params']['bidirectional'],
                        'device': device}
    #model = SpeechClassifierModel(**model_params)
    model = ConformerModel(**model_params)
    model.load_state_dict(model_state_dict)
    model.eval()
    voiced_confidences = []
    #s = socket.socket()         # Create a socket object
    host = socket.gethostname() # Get local machine name
    port = 7012                 # Reserve a port for your service.

    global wake_word_detector
    print("Waiting for the wake word to start the recording (Wake word detector):")
    newpage = work_method(stream, model, request)
    if newpage:
        return render(request, template_name='page.html')
   # return newpage

def page_method(request):
    wake_word_detector = True if request.method == 'POST' else False
    s = socket.socket()         # Create a socket object
    host = socket.gethostname() # Get local machine name
    port = 7012                 # Reserve a port for your service.
    host = 'adamantium.cse.ohio-state.edu'
    port1 = '7***'
    print("The client is pointed towards the server at the following address:") 
    print(host,port1)
    #c, addr = s.accept()
    FORMAT = pyaudio.paInt16
    CHANNELS = 1

    SAMPLE_RATE = 16000
    CHUNK = 1024 #int(SAMPLE_RATE / 10) # Still has to differntiate between various chunk sizes
    num_samples = 1536
    audio = pyaudio.PyAudio()
    stream = audio.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=SAMPLE_RATE,
                    input=True,
                    frames_per_buffer=1024)
    continue_recording = True if wake_word_detector else False
    if wake_word_detector:
        s.connect((host, port))
        print("Connected to the server")
    while continue_recording:
        audio_chunk = stream.read(1024, exception_on_overflow = False)
        try:
            s.send(audio_chunk)
            print("sent")
            received_data = s.recv(1024)
            print(received_data)

        except:
            print("Server sent a closing signal ....might be the connection was closed by the server")
            continue_recording = False
            s.close()
            stream.stop_stream()
            stream.close()
            audio.terminate()
            results = ast.literal_eval(received_data.decode('utf-8'))
            print(results)
            return render(request, 'page.html', context={'results': results})
            


def work_method(stream, model, request):
    print("The client is pointed towards the server")
    frames = []
    data = []
    res = True
    while res:
        curr_chunk = stream.read(1024, exception_on_overflow = False)
        data.append(curr_chunk)
        print("....waiting....", end='')
        wake_word_detector = process_frame(curr_chunk,frames, data, 0, 0, model, request)
        if wake_word_detector:
            return wake_word_detector
            