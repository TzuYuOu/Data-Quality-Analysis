from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name="home"),
    path('feature-transform', views.feature_transform, name="feature-transform"),
    path('download_module', views.download_module, name='download-module'),
]
