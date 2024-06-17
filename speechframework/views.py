from django.http import HttpResponse
from django.template import loader
from transformers import BertTokenizer
from django.shortcuts import render
import speechframework.audio_api as audio_api
TOK = BertTokenizer.from_pretrained('bert-base-uncased')


def home():
    """
    The home page of the website saved at main.html
    :return: The rendered template
    
    """
    template = loader.get_template('main.html')
    return HttpResponse(template.render())


def process_audio(request):
    output = audio_api.start_wake_detection(request)
    return render(request, template_name='main.html', context={'output': output})


def audio_main(request):
    """
    The initial method for the audio processing on IKLE mode from the home page
    :param request: The request object
    :return: The rendered template
    :description: This method opens the audio stream, loads the wake word model and waits for the wake word to start the recording, once detected, it renders the page.html template

    """
    audio_api.start_wake_detection(request)
    return render(request, template_name='page.html')


def page_method(request):
    """
    This method load on the page.html template
    :param request: The request object
    :return: The rendered template
    :description: Record the incoming audio, till 5 second of silence is detected, then transcribe the audio and align the transcription with the audio

    """
    results = audio_api.transcribe_audio(request)
    return render(request, 'page.html', context={'results': results})
