from django.shortcuts import render, redirect

from django.contrib.auth import authenticate,login,logout
from .forms import SignUpForm, LoginForm

from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import speech_recognition as sr
from .models import Conversation, ConversationMessage
from .utils import process_user_query, convert_text_to_speech

from pydub import AudioSegment
import tempfile


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
    
    print("My login form: ")
    print(form)
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


@csrf_exempt
def process_audio(request):
    if request.method == 'POST':
        audio_file = request.FILES.get('audio')
        if audio_file:
            try:
                # Convert audio/webm to PCM WAV format
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as temp_wav_file:
                    audio = AudioSegment.from_file(audio_file, format="webm")
                    audio.export(temp_wav_file.name, format="wav")

                    # Use the speech_recognition library
                    r = sr.Recognizer()
                    with sr.AudioFile(temp_wav_file.name) as source:
                        audio_data = r.record(source)
                        text = r.recognize_google(audio_data)

                # Assume the user is authenticated
                user = request.user

                # Get or create a new Conversation instance for the user
                conversation, created = Conversation.objects.get_or_create(user=user)

                # Create a new ConversationMessage instance for the user's query
                user_message = ConversationMessage.objects.create(
                    conversation=conversation,
                    sender='user',
                    message=text
                )

                # Generate a response
                response_text = process_user_query(text)

                # Create a new ConversationMessage instance for the bot's response
                bot_message = ConversationMessage.objects.create(
                    conversation=conversation,
                    sender='bot',
                    message=response_text
                )

                # Optionally, convert the response to speech
                audio_file_path = convert_text_to_speech(response_text)

                return JsonResponse({
                    'response': response_text,
                    'audio_file': audio_file_path
                })
            except Exception as e:
                return JsonResponse({'error': str(e)})
    return JsonResponse({'error': 'Invalid request'})