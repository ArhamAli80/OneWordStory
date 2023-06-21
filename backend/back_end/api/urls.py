from django.urls import include, path
from . import views

urlpatterns = [
    path('', views.generate_word, name='generate_word'),
]