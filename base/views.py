from django.shortcuts import render
import os
from django.http import FileResponse

# Create your views here.
def home(request):
    return render(request, 'base/home.html')


def feature_transform(request):
    return render(request, 'base/feature-transform.html')

def download_module(request):
    filename = 'media/example/HighFreqTran.rar'
    response = FileResponse(open(filename, 'rb'))
    return response