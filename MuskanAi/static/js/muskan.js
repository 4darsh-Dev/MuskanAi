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

        sideBar.style.flex = "0.7";
        sideBar.style.display = "flex";
        sideBtnCond = true;

    }

    
});