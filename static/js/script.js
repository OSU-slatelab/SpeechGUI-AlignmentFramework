document.addEventListener('DOMContentLoaded', function() {
    // Get references to the buttons and content divs
    const workModeBtn = document.getElementById('workModeBtn');
    const testModeBtn = document.getElementById('testModeBtn');
    const workModeContent = document.getElementById('workModeContent');
    const testModeContent = document.getElementById('testModeContent');
    
    //console.log({ results })

    // Event listeners for the buttons
    workModeBtn.addEventListener('click', function() {
        const csrftoken = getCookie('csrftoken');
        const kwd = document.getElementById('wake-word')
        
            fetch('/audio_main/', {
                method: 'POST',
                headers: {
                    'X-CSRFToken': csrftoken
                },
                
                body: JSON.stringify({ kwd: kwd.value })
            })
            
            .then(response => response.text())
            .then(data => {
                console.log("Received response from backend:", data);
                const responseContainer = document.getElementById('responseContainer');
                var queryParams = '?stream=value1&param2=value2';
                var pageUrl = '/page/' + queryParams; 
                window.location.href = pageUrl;
                //responseContainer.innerHTML = data;
                //$('#result-container').html(data);
                // Handle response from backend (e.g., display results)
            })
            .catch(error => {
                console.error("Error sending audio data to backend:", error);
            });
            workModeContent.style.display = 'block';
            testModeContent.style.display = 'none';
        
    });

    testModeBtn.addEventListener('click', function() {
        workModeContent.style.display = 'none';
        testModeContent.style.display = 'block';
    });
});
// Define variables to hold the recording stream and media recorder
let audioStream;
let mediaRecorder;

// Function to start recording audio
function startRecording() {
    navigator.mediaDevices.getUserMedia({ audio: true })
        .then(stream => {
            audioStream = stream;
            mediaRecorder = new MediaRecorder(stream);
            mediaRecorder.start();
            console.log("Recording started...");
        })
        .catch(error => {
            console.error("Error accessing microphone:", error);
        });
}
function getCookie(name) {
    const cookieValue = document.cookie.match('(^|;)\\s*' + name + '\\s*=\\s*([^;]+)');
    return cookieValue ? cookieValue.pop() : '';
}
// Function to stop recording audio and send it to the backend
function stopRecording() {
    if (mediaRecorder && mediaRecorder.state === "recording") {
        mediaRecorder.stop();
        console.log("Recording stopped...");

        // Retrieve recorded audio data
        mediaRecorder.ondataavailable = event => {
            const audioBlob = event.data;
            
            // Convert audio blob to a file
            const audioFile = new File([audioBlob], 'recording.wav', { type: audioBlob.type });
            
            // Send audio file to backend (replace 'upload-audio' with your backend endpoint)
            const formData = new FormData();
            formData.append('audio', audioFile);
            const csrftoken = getCookie('csrftoken');
            fetch('/process_audio/', {
                method: 'POST',
                headers: {
                    'X-CSRFToken': csrftoken
                },
                body: JSON.stringify({ kwd: kwd.value })
            })
            .then(response => response.json())
            .then(data => {
                console.log("Received response from backend:", data);
                // Handle response from backend (e.g., display results)
            })
            .catch(error => {
                console.error("Error sending audio data to backend:", error);
            });
        };
    }
}

// Event listener for the "Start Recording" button
document.getElementById('stop-btn').addEventListener('click', function(){
    var queryParams = '?stream=value1&param2=value2';
    var pageUrl = '/page/' + queryParams; 
    window.location.href = pageUrl;
});
document.getElementById('startbtn').addEventListener('click', function(){
    const csrftoken = getCookie('csrftoken');
    kwd = document.getElementById('key-word')
    fetch('/process_audio/', {
        method: 'POST',
        headers: {
            'X-CSRFToken': csrftoken
        },
        
        body: JSON.stringify({ kwd: kwd.value, res:true  })
    })
    
    .then(response => response.text())
    .then(data => {
        console.log("Received response from backend:", data);
        //const responseContainer = document.getElementById('responseContainer');
        //var queryParams = '?stream=value1&param2=value2';
        //var pageUrl = '/page/' + queryParams; 
        //window.location.href = pageUrl;
        //responseContainer.innerHTML = data;
        //$('#result-container').html(data);
        // Handle response from backend (e.g., display results)
    })
    .catch(error => {
        console.error("Error sending audio data to backend:", error);
    });
});
document.getElementById('stopbtn').addEventListener('click', function(){
    const csrftoken = getCookie('csrftoken');
    kwd = document.getElementById('key-word')
    fetch('/process_audio/', {
        method: 'POST',
        headers: {
            'X-CSRFToken': csrftoken
        },
        
        body: JSON.stringify({ kwd: kwd.value, res:false  })
    })
    
    .then(response => response.text())
    .then(data => {
        console.log("Received response from backend:", data);
        const testModeContent = document.getElementById('testModeContent');
    
        testModeContent.style.display = 'block';
        document.body.innerHTML = data
        })
    .catch(error => {
        console.error("Error in stop button:", error);
    });
    
});

// Event listener for the "Stop Recording" button

//document.getElementById('stop-btn').addEventListener('click', stopRecording);
