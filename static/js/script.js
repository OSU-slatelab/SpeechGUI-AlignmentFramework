document.addEventListener('DOMContentLoaded', function() {
    // Get references to the buttons and content divs
    const workModeBtn = document.getElementById('workModeBtn');
    const testModeBtn = document.getElementById('testModeBtn');
    const workModeContent = document.getElementById('workModeContent');
    const testModeContent = document.getElementById('testModeContent');
    
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

function getCookie(name) {
    const cookieValue = document.cookie.match('(^|;)\\s*' + name + '\\s*=\\s*([^;]+)');
    return cookieValue ? cookieValue.pop() : '';
}


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
