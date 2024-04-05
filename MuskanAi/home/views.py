from django.shortcuts import render, redirect

from django.contrib.auth import authenticate,login,logout
from .forms import SignUpForm, LoginForm

# Create your views here.

def index(request):
    return render(request, "index.html")

def about(request):
    return render(request, "about.html")

#login page
def loginUser(request):
    if request.method == "POST":
        form = LoginForm(request.POST)
        if form.is_valid():
            uname = form.cleaned_data['username']
            pwd = form.cleaned_data['password']
            user = authenticate(username=uname, password=pwd)
            if user is not None:
                login(request, user)
                return redirect("muskan")
    else:
        form = LoginForm()
    return render(request, "login.html", {'form':form})



def logoutUser(request):
    logout(request)
    return redirect("home")

# signup page
def signupUser(request):
    if request.method == "POST":
        form = SignUpForm(request.POST)
        if form.is_valid():
            form.save()
            return redirect('muskan')
    else:
        form = SignUpForm()
    
    return render(request, "signup.html", {'form':form})



# official chat page
def muskan(request):
    return render(request, "muskan.html")
