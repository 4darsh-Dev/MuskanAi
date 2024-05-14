# Utils file

from django.contrib.auth.models import User
from .models import Conversation, ConversationMessage

from gtts import gTTS
import os


def process_user_query(user, query):
    # Get or create a new Conversation instance for the user
    conversation, created = Conversation.objects.get_or_create(user=user)

    # Create a new ConversationMessage instance for the user's query
    user_message = ConversationMessage.objects.create(
        conversation=conversation,
        sender='user',
        message=query
    )

    # Generate a response (replace this with your actual logic)
    response = f"This is a response to your query: {query}"

    # Create a new ConversationMessage instance for the bot's response
    bot_message = ConversationMessage.objects.create(
        conversation=conversation,
        sender='bot',
        message=response
    )

    return response


def convert_text_to_speech(text):
    # Create a gTTS object
    tts = gTTS(text=text, lang='en')

    # Save the speech as an audio file
    audio_file_path = os.path.join("MuskanAi/static/audio", f'response.mp3')
    tts.save(audio_file_path)

    return audio_file_path

