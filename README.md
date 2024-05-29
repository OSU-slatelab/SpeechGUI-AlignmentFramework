# SpeechGUI-AlignmentFramework
### Abstract
Speech is the new essential fuel for human-computer interaction. With the current trend of
modern-day human-computer interaction and its increasing reliance on voice commands,
the development of a robust speech framework is paramount. The demand for intuitive
interfaces capable of comprehending and responding to natural language commands has
considerably increased. When interacting across various domains, lack of synchronization
between different user inputs can cause confusion and dissatisfaction, which can lead to
disjointed user experience and lost productivity. This project presents a precisely built
"Robust Speech-GUI Integration Framework for Frontend Audio Detection and Tracking".
This framework facilitates the conversion of speech commands to text and precisely aligns
them with the corresponding Graphical User Interface (GUI) events. The framework stands
out as a sophisticated solution with a plug-and-play architecture due to its integration of
multiple methodologies for each of its submodules. The initial module for speech command
onset detection features three variations: push-to-talk, predefined, and customized wake
word. Subsequently, the Silero VAD off-the-shelf model is utilized to ascertain the
endpoint of speech. Following this, the WhisperX module provides precise word-level
transcription, meticulously timestamped to align with concurrently captured GUI events.
The system demonstrates robustness, offering full functionality either on-device or
partially in the cloud via socket communication.

### Installation
To successfully install the environment needed to run this code, run the following commands:
```
conda create --name speechgui --file requirements_versionless.txt
conda activate speechgui
conda install pytorch==2.0.0 torchaudio==2.0.0 pytorch-cuda=11.8 -c pytorch -c nvidia
pip install git+https://github.com/m-bain/whisperx.git
pip install conformer
```

### Running the Program
Start the server by running:
```
python manage.py runserver
```
You should then receive a message that says something along the lines of:
```
Starting development server at http://127.0.0.1:8000/
```
Open the provided web address in your browser. You can then select a wake word and choose whether you want to enter test mode or ICICLE mode.

### Further Project Details
[Detailed project report online version](https://drive.google.com/file/d/1E15Q1RK25Z6BZQgCl_zwUFvqaYhEgOtn/view?usp=sharing)

##### Project Report Copyrighted by Beulah Karrolla 2024
