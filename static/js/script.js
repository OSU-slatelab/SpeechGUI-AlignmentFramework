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

function getCookie(name) {
    const cookieValue = document.cookie.match('(^|;)\\s*' + name + '\\s*=\\s*([^;]+)');
    return cookieValue ? cookieValue.pop() : '';
}


// Event listener for the "Start Recording" button
document.getElementById('stop-btn').addEventListener('click', function(){
    var queryParams = '?stream=value1&param2=value2';
    var pageUrl = '/page/' + queryParams; 
    window.location.href = pageUrl;
});


// Event listener for the "Stop Recording" button

//document.getElementById('stop-btn').addEventListener('click', stopRecording);
