document.addEventListener('DOMContentLoaded', function () {
    const micButton = document.getElementById('mic-button');
    const crossButton = document.getElementById('cross-button');
    const transcript = document.getElementById('transcript');
    const chatInterface = document.getElementById('chat-interface');
    
    micButton.addEventListener('click', function () {
        // Code to handle microphone button click event
        // You can add logic to start/stop voice recognition here
    });
    
    crossButton.addEventListener('click', function () {
        // Code to handle cross button click event
        // You can add logic to clear the transcript here
    });
    
    // Function to add a new message to the chat interface
    function addMessage(message, sender) {
        const messageDiv = document.createElement('div');
        messageDiv.classList.add('chat-message');
        messageDiv.textContent = `${sender}: ${message}`;
        chatInterface.appendChild(messageDiv);
    }
    
    // Example usage:
    addMessage('Hello!', 'User');
    addMessage('Hi there!', 'Assistant');
});


// toggle side bar button

let sideBtn = document.getElementById('side-button');

sideBtnCond  = true;

sideBtn.addEventListener('click', () => {
    let sideBar = document.getElementById('assistant');
    if(sideBtnCond)
    {
        sideBar.style.flex = "0";
        sideBar.style.display = "none";
        sideBtnCond = false;
    }
    else
    {

        sideBar.style.flex = "0.5";
        sideBar.style.display = "flex";
        sideBtnCond = true;

    }

    
});


// JavaScript
let mediaRecorder;
let recordedChunks = [];
let isRecording = false;
let pauseTimeout;

const startConversationButton = document.getElementById('startConversationButton');
const statusElement = document.getElementById('status');

startConversationButton.addEventListener('click', async () => {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    mediaRecorder = new MediaRecorder(stream);

    mediaRecorder.addEventListener('dataavailable', (event) => {
      recordedChunks.push(event.data);
    });

    mediaRecorder.addEventListener('stop', () => {
      const audioBlob = new Blob(recordedChunks, { type: 'audio/webm' });
      const formData = new FormData();
      formData.append('audio', audioBlob, 'recording.webm');

      // Send the recorded audio to the backend
      fetch('/process-audio/', {
        method: 'POST',
        body: formData,
      })
        .then((response) => response.json())
        .then((data) => {
          // Handle the response from the backend
          console.log(data);
          statusElement.textContent = 'Received response. You can speak again.';
          isRecording = false;
          startRecording();
        })
        .catch((error) => {
          console.error('Error:', error);
          statusElement.textContent = 'Error occurred. Please try again.';
          isRecording = false;
          startRecording();
        });
    });

    startRecording();
    startConversationButton.disabled = true;
  } catch (err) {
    console.error('Error:', err);
    statusElement.textContent = 'Error occurred. Please try again.';
  }
});

function startRecording() {
  if (!isRecording) {
    isRecording = true;
    recordedChunks = [];
    mediaRecorder.start();
    statusElement.textContent = 'Recording...';

    pauseTimeout = setTimeout(() => {
      mediaRecorder.stop();
      statusElement.textContent = 'Processing...';
    }, 2000);
  }
}

function stopConversation() {
  if (mediaRecorder && mediaRecorder.state !== 'inactive') {
    mediaRecorder.stop();
  }
  startConversationButton.disabled = false;
  statusElement.textContent = 'Conversation ended.';
}

// Add event listener for stopping the conversation (e.g., pressing Esc key)
document.addEventListener('keydown', (event) => {
  if (event.key === 'Escape') {
    stopConversation();
  }
});