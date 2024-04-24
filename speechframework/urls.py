'''from django.urls import path
from speechframework import views
from speechframework.views import audio_main, process_audio, process_frame, save_file
from .views import process_audio 

urlpatterns = [
    path("", views.home, name="home"),
    path('process_audio/', process_audio, name='process_audio'),
    path('process-frame/', views.audio_main, name='audio_main'),
]

from django.views.generic.base import TemplateView
from django.contrib import admin
from django.urls import path
from django.conf import settings
from django.conf.urls.static import static'''


'''urlpatterns = [
    path('admin/', admin.site.urls),
    path('', TemplateView.as_view(template_name='HomePage.html')),
    path('results/', TemplateView.as_view(template_name='Results.html')),
    path('predict-dicom/', DiagnoseDICOMImage.as_view()),
    path('predict-mammogram/', DiagnoseMammogramImage.as_view()),
    path('predict-lung-image/', DiagnoseLungXRay.as_view()),
] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)'''
urlpatterns = []