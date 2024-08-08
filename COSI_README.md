# COSI Wake Gesture Setup

### Description
This code allows for wake gesture detection for the COSI chatbot system through the use of client-server communication. The client side features a front-end system that captures video input from a webcam on a local device using CV2 video capture and sends each video frame to the server back-end for processing using a websockets connection. After receiving a video frame, the server features back-end processing that detects each facial ID within the frame using the YOLOV8 image segmentation model, creates a facial mesh and hand landmarks for each ID using Mediapipe, and outputs facial engagement and gesture recognition. Using this output, the code checks for a wake gesture to start audio transcription using one of the following recognized gestures:

<ul>
  <li>Closed_Fist</li>
  <li>Open_Palm</li>
  <li>Pointing_Up</li>
  <li>Thumb_Up</li>
  <li>Thumb_Down</li>
  <li>Victory</li>
  <li>ILoveYou</li>
</ul>

If the wake gesture is detected, audio input is taken in from the microphone from the local device and processed on the client front-end for transcription using Whisper. 

### VirtualBox VM
In order to capture audio input, a VirtualBox VM with Ubuntu must be utilized due to WSL failing to capture microphone input appropriately.

[Download Ubuntu Desktop ISO image](https://ubuntu.com/download/desktop)

[Follow instructions for VM creation](https://ubuntu.com/tutorials/how-to-run-ubuntu-desktop-on-a-virtual-machine-using-virtualbox#1-overview)

Download VirtualBox extension to allow for webcam access and audio input.
        <ol>
            <li>[Extension download](https://www.virtualbox.org/wiki/Downloads)</li>
            <li>Extension can be accessed within the Oracle VM VirtualBox Manager in Files > Tools > Extension Pack Manager</li>
            <li>Within the VM, go to Devices > Webcams and turn on the desired webcam and go to Devices > Audio and turn on Audio Input.</li>
        </ol>

[(Mostly) accurate instructions from Medium](https://medium.com/nerd-for-tech/how-to-use-camera-in-windows-10-virtualbox-aa92ffbe1f24)
### Conda Environment 
Install Conda within the Ubuntu VM to allow for environment setup. 
To successfully install the client environment needed to run this code, run the following commands:
```
sudo apt-get install libasound-dev
wget https://files.portaudio.com/archives/pa_stable_v190700_20210406.tgz
tar -xvzf pa_stable_v190700_20210406.tgz
cd portaudio
./configure
make
sudo make install
sudo ldconfig
```
This will install portaudio. You need port audio in order to successfully install PyAudio. You can then proceed with installing python requirements. Start by install PyTorch first. Note: You must use 2.0.0.
```
conda create --name speechgui --file client_audio_requirements_versionless.txt
conda activate speechgui
pip install torch==2.0.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```
Once PyTorch is installed you can let pip handle the rest. You can use conda if you want, but do not install PyAudio with Conda. There is a known issue with the default package build that conda selects for 0.2.14. Use the default package from pip to avoid this.
```
pip install -r client_audio_requirements.txt
pip install -r client_video_requirements.txt
```
Your client environment should now be set up. For the server environment, run `pip install -r

### VSCode Installation 
Download Debian file for [VSCode in Ubuntu](https://code.visualstudio.com/download) and then run the following commnands:
```
cd /Downloads
sudo apt install ./{code_filename}.deb
sudo apt install apt-transport-https
sudo apt update
sudo apt install code
```
To run VSCode, use the `command code --password-store="basic"` - running `code` results in VSCode not responding error otherwise.

### Running the Program
First, establish port forwarding between the client and server using the following command:
```
ssh -L [local_address]:[remote_server_port] [username]@[remote_server]
```
Within the code currently, the local address will be 5000:localhost and the remote server port will be 8000. Once the port forwarding is established, run the server.py program on the server and wait for "Server waiting..." to be outputted within the terminal. Then, run the following command where "gesture" represents the one of the recognized Mediapipe gestures to be used as a wake gesture:
```
python client.py gesture 
```
