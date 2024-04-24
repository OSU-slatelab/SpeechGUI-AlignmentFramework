
document.addEventListener('DOMContentLoaded', function() {
    var touchPath = [];

    // Start monitoring mousemove events on the container element
    var container = document.getElementById('touchPathContainer');
    container.addEventListener('mousemove', function(event) {
        touchPath.push({ x: event.offsetX, y: event.offsetY+100, time:event.timeStamp });
    
        //touchPath.push({ x: event.clientX, y: event.clientY, time:event.timeStamp });
    });

    function extractSegments(data, limx, limy) {
        const segments = [];
        let segment = [];
        for (const point of data) {
            if (limx <= point.time && point.time <= limy) {
                segment.push(point);
            } else {
                if (segment.length > 0) {
                    segments.push(segment);
                    segment = [];
                }
            }
        }
        if (segment.length > 0) {
            segments.push(segment);
        }
        return segments;
    }

    // Function to draw path from stored mouse positions
    function drawPath(tPath) {
        const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
        svg.setAttribute('width', '100%');
        svg.setAttribute('height', '100%');
        
        svg.style.position = 'absolute';
        svg.style.top = '250px';
        svg.style.left = '20px';
        //svg.style.zIndex = '1';
        svg.style.visibility = 'visible'; // Set visibility to visible

        const path = document.createElementNS('http://www.w3.org/2000/svg', 'path');
        let pathString = 'M ' + tPath[0].x + ' ' + tPath[0].y + ' ';
        for (let i = 1; i < tPath.length; i++) {
            pathString += 'L ' + tPath[i].x + ' ' + tPath[i].y + ' ';
        }
        path.setAttribute('d', pathString);
        path.setAttribute('fill', 'none');
        path.setAttribute('stroke', 'blue');
        path.setAttribute('stroke-width', '2');
        svg.appendChild(path);
        
        var container = document.getElementById('touchPathContainer');
        //container.style.position = 'absolute'
        container.appendChild(svg);
    }
    

    // Add onclick event listeners to buttons
    var buttons = document.querySelectorAll('.button');
    buttons.forEach(function(button) {
        validJsonString = button.id.replace(/'/g, '"');
        buttonData = JSON.parse(validJsonString);
        limx = buttonData['start']
        limy = buttonData['end']
        button.addEventListener('click', function() {
            
          path = extractSegments(touchPath, limx=limx*1000, limy=limy*1000)
            drawPath(path[0]);
        });
    });       
    const csrftoken = getCookie('csrftoken');
        fetch('/page_method/', {
            method: 'POST',
            headers: {
                'X-CSRFToken': csrftoken
            },
            body: ''
        })
        
        .then(response => response.text())
        .then(data => {
            //window.location.reload();
            // Parse the JSON data to JavaScript object
            //const results = JSON.parse(results);
            // Do something with the results
            //console.log(results);
        //.then(data => {})
            console.log("Received response from backend:", data);
            this.documentElement.innerHTML = data;
            buttons = document.querySelectorAll('.button');
            buttons.forEach(function(button) {
                button.addEventListener('click', function() {
                    validJsonString = button.id.replace(/'/g, '"');
                    buttonData = JSON.parse(validJsonString);
                    limx = buttonData['start']
                    limy = buttonData['end']

                    path = extractSegments(touchPath, limx=limx*1000, limy=limy*1000)
                    drawPath(path[0]);
                });
            }); 
            //var results = JSON.parse('{{ results | safe }}');
            // Now you can access the 'results' variable in JavaScript
            //console.log(results);
            const responseContainer = document.getElementById('result-container');
            //window.location.href = '/main/';
            //responseContainer.innerHTML = data;
            //$('#result-container').html(data);
            // Handle response from backend (e.g., display results)
        })
        .catch(error => {
            console.error("Error sending audio data to backend:", error);
        });
});
function getCookie(name) {
    const cookieValue = document.cookie.match('(^|;)\\s*' + name + '\\s*=\\s*([^;]+)');
    return cookieValue ? cookieValue.pop() : '';
}

