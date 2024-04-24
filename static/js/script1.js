function openTab(tabName) {
    var i, tabContent;

    // Hide all tab content
    tabContent = document.getElementsByClassName('tab-content');
    for (i = 0; i < tabContent.length; i++) {
        tabContent[i].style.display = 'none';
    }

    // Show the selected tab content
    document.getElementById(tabName).style.display = 'block';
}
// Function to handle speech recognition
function startSpeechRecognition() {
    // Initialize SpeechRecognition object
    const recognition = new webkitSpeechRecognition();
    recognition.continuous = true;
    recognition.interimResults = true;

    // Event handler when speech is recognized
    recognition.onresult = function(event) {
        const transcript = event.results[event.results.length - 1][0].transcript;
        outputDiv.textContent = transcript;
    };

    // Event handler when speech recognition ends
    recognition.onend = function() {
        // Restart recognition if needed
        recognition.start();
    };

    // Start speech recognition
    recognition.start();
}

// Event handler when user clicks the start button for speech recognition
startSpeechBtn.addEventListener('click', startSpeechRecognition);

// Function to continuously capture microphone input and send it to the backend
function captureAndSendAudio() {
    navigator.mediaDevices.getUserMedia({ audio: true })
        .then(function(stream) {
            // Initialize the MediaRecorder to capture microphone input
            var mediaRecorder = new MediaRecorder(stream);
            var chunks = [];

            // Start recording when the stream is available
            mediaRecorder.start();

            // Event handler for when data is available
            mediaRecorder.ondataavailable = function(event) {
                
                chunks.push(event.data);
                fetch('/process-audio/', {
                    method: 'POST',
                    body: formData
                })
            };

            // Event handler for when recording is stopped
            mediaRecorder.onstop = function() {
                // Concatenate the recorded audio chunks into a single Blob
                var audioBlob = new Blob(chunks, { type: 'audio/wav' });

                // Send the audio data to the backend for processing
                var formData = new FormData();
                formData.append('audio', audioBlob);

                fetch('/process-audio/', {
                    method: 'POST',
                    body: formData
                })
                .then(response => response.json())
                .then(data => {
                    // Handle the response from the backend
                    console.log(data);
                });
            };
        })
        .catch(function(error) {
            console.error('Error accessing microphone:', error);
        });
}

// Call the function to start capturing and sending audio when the page loads
window.onload = function() {
    fetch('/process_frame/')
        .then(response => response.json())
        .then(data => {
            // Handle the response from the server
            console.log(data);
        })
        .catch(error => {
            console.error('Error calling models method:', error);
        });
};
