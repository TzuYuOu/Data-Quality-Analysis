from django.shortcuts import render, redirect
from django.contrib.auth import authenticate, login, logout
from django.contrib import messages
from django.contrib.auth.forms import UserCreationForm
# Create your views here.

def login_user(request):
	if request.method == "POST":
		username = request.POST['username']
		password = request.POST['password']
		user = authenticate(request, username=username, password=password)
		if user is not None:
			login(request, user)
			return redirect('list-data')
		else:
			messages.success(request, ("There was an error logging in ! Try again..."))
			return redirect('login')

	return render(request, 'authentication/login.html')

def register_user(request):
		if request.method == "POST":
				form = UserCreationForm(request.POST)
				if form.is_valid():
						form.save()
						username = form.cleaned_data['username']
						password = form.cleaned_data['password1']
						user = authenticate(username=username, password=password)
						login(request, user)
						messages.success(request, "Registeration successful!")
						return redirect('list-data')
		else:
  			form = UserCreationForm()
		
		context = {'form': form}
		return render(request, 'authentication/register_user.html', context)

def logout_user(request):
	logout(request)
	messages.success(request, ("You were logged out!"))
	return redirect('home')