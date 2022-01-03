from django.shortcuts import render
import os

# Create your views here.
def home(request):
    return render(request, 'base/home.html')
