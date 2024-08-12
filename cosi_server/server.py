import cv2
import asyncio
import websockets
import base64
import json
import torch
import pyaudio
import whisperx
import wave
import numpy as np
import mediapipe as mp

from ultralytics import YOLO
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

FORMAT = pyaudio.paInt16
CHANNELS = 1
SAMPLE_RATE = 16000
model_asr = whisperx.load_model("small", 'cpu', compute_type="int8", language="en")
whisperx_align, metadata = whisperx.load_align_model(language_code="en", device="cpu")
model_vad, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                                model='silero_vad',
                                force_reload=True)

class Mp_Hands():
    def __init__(self) -> None:
        # Sets the hand detection and hand landmark points.
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.mp_hands = mp.solutions.hands    
        self.Hands = self.mp_hands.Hands(model_complexity=0,min_detection_confidence=0.5,min_tracking_confidence=0.5)  
        # Sets the gesture recognition.
        self.base_options = python.BaseOptions(model_asset_path='gesture_recognizer.task')
        self.options = vision.GestureRecognizerOptions(base_options=self.base_options)
        self.recognizer = vision.GestureRecognizer.create_from_options(self.options)

    def get_hands(self, image):
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_2 = image.copy()
        results = self.Hands.process(image)

        # Draw the hand annotations on the image.
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(
                    image,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing_styles.get_default_hand_landmarks_style(),
                    self.mp_drawing_styles.get_default_hand_connections_style())   
                
        image_3 = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_2)

        recognition_result = self.recognizer.recognize(image_3)

        if recognition_result.gestures: 
            top_gesture = recognition_result.gestures[0][0]

            if top_gesture.score > 0.5:
                cv2.putText(image,top_gesture.category_name,(200,50),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)

            gesture = top_gesture.category_name
            score = top_gesture.score

            return image, gesture, score 
    
        else:
            gesture = "N/A"
            score = "N/A"
    
            return image, gesture, score 
        # return image


class Mp_Face_Detection():
    '''
    Class Contains all elements to detect a face, plot face points landmarks, and estimate head orientation.
    '''
    def __init__(self) -> None:
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(min_detection_confidence=0.5,min_tracking_confidence=0.5, max_num_faces = 1)
        self.mp_drawing = mp.solutions.drawing_utils
        self.drawing_spec = self.mp_drawing.DrawingSpec(color=(128,0,128),thickness=1,circle_radius=1)
        
    def get_mesh(self, image):
        results = self.face_mesh.process(image)
        
        return results
    
    def get_2D_3D(self,image, results):
        img_h , img_w, img_c = image.shape
        face_2d = []
        face_3d = []
        text = "N/A"

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                for idx, lm in enumerate(face_landmarks.landmark):
                    if idx == 33 or idx == 263 or idx ==1 or idx == 61 or idx == 291 or idx==199:
                        if idx ==1:
                            nose_2d = (lm.x * img_w,lm.y * img_h)
                            nose_3d = (lm.x * img_w,lm.y * img_h,lm.z * 3000)
                        x,y = int(lm.x * img_w),int(lm.y * img_h)

                        face_2d.append([x,y])
                        face_3d.append(([x,y,lm.z]))


                #Get 2d Coord
                face_2d = np.array(face_2d,dtype=np.float64)

                face_3d = np.array(face_3d,dtype=np.float64)

                focal_length = 1 * img_w

                cam_matrix = np.array([[focal_length,0,img_h/2],
                                    [0,focal_length,img_w/2],
                                    [0,0,1]])
                distortion_matrix = np.zeros((4,1),dtype=np.float64)

                success,rotation_vec,translation_vec = cv2.solvePnP(face_3d,face_2d,cam_matrix,distortion_matrix)


                #getting rotational of face
                rmat,jac = cv2.Rodrigues(rotation_vec)

                angles,mtxR,mtxQ,Qx,Qy,Qz = cv2.RQDecomp3x3(rmat)

                x = angles[0] * 360
                y = angles[1] * 360
                z = angles[2] * 360

                #here based on axis rot angle is calculated
                if y < -10:
                    # text="Left"
                    text="Right"

                elif y > 10:
                    # text="Right"
                     text="Left"

                elif x < -10:
                    text="Down"
                elif x > 10:
                    text="Up"
                else:
                    text="Engaged"

                nose_3d_projection,jacobian = cv2.projectPoints(nose_3d,rotation_vec,translation_vec,cam_matrix,distortion_matrix)

                p1 = (int(nose_2d[0]),int(nose_2d[1]))
                p2 = (int(nose_2d[0] + y*10), int(nose_2d[1] -x *10))

                cv2.line(image,p1,p2,(255,0,0),3)

                if text == "Engaged":
                    cv2.putText(image,text,(20,50),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
                else:
                    cv2.putText(image, text,(20,50),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2)

                cv2.putText(image,"x: " + str(np.round(x,2)),(500,50),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)
                cv2.putText(image,"y: "+ str(np.round(y,2)),(500,100),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)
                cv2.putText(image,"z: "+ str(np.round(z, 2)), (500, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                
            self.mp_drawing.draw_landmarks(image=image,
                                  landmark_list=face_landmarks,
                                  connections= self.mp_face_mesh.FACEMESH_CONTOURS,
                                  landmark_drawing_spec= self.drawing_spec,
                                  connection_drawing_spec= self.drawing_spec)

                
        return image, text

class ID_Person ():
    '''
    Class encapsulates the Mp Hands and Face Detection in a single class,
    so a person detected a in a frame has it's own classifiers. 
    '''
       
    def __init__(self, id) -> None:
        self.mesh_detector = Mp_Face_Detection()
        self.mp_hands_detector = Mp_Hands()
        self.id = id
                
    def get_mesh(self, person_image):
        img = cv2.cvtColor(person_image,cv2.COLOR_BGR2RGB)
        img.flags.writeable = False
        results = self.mesh_detector.get_mesh(img)
        img.flags.writeable = True
        img = cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
        text = 'N/A'
        try:
            img, text = self.mesh_detector.get_2D_3D(img, results)
            return img, text
        except:
            return img, text      
         
    def get_hands(self, person_image):
        img, gesture, score = self.mp_hands_detector.get_hands(person_image)
        return img, gesture, score
    
class Global_View():
    '''
    Class keeps tracks of all people within a frame through a dictionary of ID_Person classes.
    '''
    def __init__(self) -> None:
        self.ids_history = {}
        self.people_track = {}
    
    #Add a person to the track if it doesn't exist already.
    def add_person(self, person_id):
        if (self.add_track_id(person_id)):
            self.people_track[person_id] = ID_Person(person_id)

    #It checks whether the ID already exists.
    #If it does exist it keeps count how many times it has been obersved through the video.
    #This feature was added for future work.
    def add_track_id(self, track_id ):
        Not_exist_flag= True
        if track_id in self.ids_history :
            self.ids_history[track_id] += 1
            Not_exist_flag = False
            
        else:
            self.ids_history[track_id] = 1  
        
        return Not_exist_flag
    
def plot_object_id(dets, image):
    '''
    It plots the ID and confidence to the detected person. 
    '''
    if len(dets) == 0:
        return image
    if isinstance(image, str):
        image = cv2.imread(image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    x1, y1, x2, y2 = dets['bbox']
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
    confidence_color = (0, 0, 255)  # Green color for confidence box
    id_color = (255, 0, 0)  # Blue color for ID box
   # Display confidence as colored box above the bounding box
    conf_text = f'Conf: {dets["conf"]:.2f}'
    conf_box_width = len(conf_text) * 20
    cv2.rectangle(image, (x1, y1 - 20), (x1 + conf_box_width, y1 - 2), confidence_color, -1)
    cv2.putText(image, conf_text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 1)

    # Display ID as colored box above the bounding box
    id_text = f'ID: {dets["id"]}'
    id_box_width = len(id_text) * 20
    cv2.rectangle(image, (x2 - id_box_width, y1 - 20), (x2, y1 - 2), id_color, -1)
    cv2.putText(image, id_text, (x2 - id_box_width, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 1)

    return image
 
def get_image(x1, y1, x2, y2, image):
    '''
    It crops an image based on the specifications, and returns this new image.
    '''
    # Ensure x2 > x1 and y2 > y1
    x1, x2 = min(x1, x2), max(x1, x2)
    y1, y2 = min(y1, y2), max(y1, y2)

    # Extract the region from the image using array slicing
    cropped_region = image[y1:y2, x1:x2]
    
    return cropped_region

def save_file(data):
    """
    Saves the audio file to the disk
    :param data: List of audio data chunks
    :return: The name of the saved file
    
    """
    output_file = "my_voice.wav"
    sample_width = 2  # 2 bytes for 16-bit audio, 1 byte for 8-bit audio
    with wave.open(output_file, 'wb') as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(sample_width)
        wf.setframerate(SAMPLE_RATE)

        for chunk in data:
            wf.writeframes(chunk)
        wf.close()
    return output_file


async def process_input(websocket):
    print("Connection established")
    count = 0
    audio_data = []
    while True:
        try: 
            async for message in websocket: 
                data = json.loads(message)

                if data["data_type"] == "video":
                    print("Received video frame")
                    frame_data = data["frame"]

                    frame = np.frombuffer(base64.b64decode(frame_data), dtype=np.uint8)
                    frame = cv2.imdecode(frame, cv2.IMREAD_COLOR)

                    mp_hands_detector = Mp_Hands()
                    mp_face_detector = Mp_Face_Detection()

                    Global_Track = Global_View()

                    device = 'cuda' if torch.cuda.is_available() else 'cpu'
                    model = YOLO('yolov8n-pose.pt').to(device)

                    results = model.track(frame, persist=True, conf = 0.5)

                    engagements, engagement_scores, gestures, gesture_scores, ages, age_scores = [], [], [], [], [], []

                    if results[0].boxes.id is not None: 
                        boxes = results[0].boxes.xywh.cpu()
                        confs = results[0].boxes.conf.cpu()
                        track_ids = results[0].boxes.id.int().cpu().tolist()

                        original_image = frame

                        for box, conf, track_id in zip(boxes,confs,track_ids):
                            Global_Track.add_person(track_id)
                            feat_box = {}
                            xc, yc, w, h = map(int, box)
                            x1 , y1 = xc-w/2, yc-h/2
                            x2, y2 = xc+w/2, yc+h/2
                            x1, y1, x2, y2 = round(x1), round(y1), round(x2), round(y2)
                            feat_box['bbox'] = [x1, y1, x2, y2]
                            feat_box['conf'] = conf
                            feat_box['id'] = track_id

                            #Feed each cropped image to its corresponding classifiers to retrieve face and 
                            #hand landmark points.         
                            cropped_image = get_image(x1, y1, x2, y2, image=original_image)

                            mesh_image, engagement = Global_Track.people_track[track_id].get_mesh(cropped_image)
                            engagements.append(engagement)
                            engagement_scores.append(conf.item())

                            mesh_image, gesture, gesture_score = Global_Track.people_track[track_id].get_hands(mesh_image)
                            gestures.append(gesture)
                            gesture_scores.append(gesture_score)

                            #Insert edited image back to original image.            
                            original_image[y1:y2, x1:x2] = mesh_image

                            #Add confidence and ID to the detected person.
                            output_frame = plot_object_id(feat_box,original_image)

                    else:
                        output_frame = frame
                        gestures.append("N/A")
                        gesture_scores.append("N/A")
                        engagements.append("N/A")
                        engagement_scores.append("N/A")

                    # Encode frame as JPEG
                    _, buffer = cv2.imencode('.jpg', output_frame)
                    output_frame_data = base64.b64encode(buffer).decode('utf-8')

                    # Stores frame and gesture data into json to send back to client
                    payload = json.dumps({'frame': output_frame_data, 'gesture': gestures, 'gesture_score': gesture_scores, 'engagement': engagements, 'engagement_score': engagement_scores})

                    # Sends frame to the client
                    await websocket.send(payload)
                    print("Sent video frame")

                elif data["data_type"] == "audio":
                    print("Received audio")
                    audio_chunk = data['audio_chunk']
                    audio_data.append(audio_chunk)

                    if data["finished"] == True:
                        audio_data = [base64.b64decode(encoded_audio) for encoded_audio in audio_data]

                        print(str(count) + " milliseconds elapsed during waiting")
                        res = save_file(audio_data)
                        continue_recording = False
                        # Transcribe the audio and align the transcription with the audio
                        audio = whisperx.load_audio(res)
                        result = model_asr.transcribe(audio, 2)
                        result2 = whisperx.align(result["segments"], whisperx_align, metadata, audio, 'cpu', return_char_alignments=False)
                        print(result["segments"][0]["text"]) if result["segments"] else print("No speech detected")
                        print(result2["word_segments"])
                        continue_recording = False
                        results = result2["word_segments"]
                        
                        if result["segments"]:
                            payload = json.dumps({"transcribed_audio": result["segments"][0]["text"]})
                        else:
                            payload = json.dumps({"transcribed_audio": "No speech detected"})

                        # Sends transcribed audio to the client
                        await websocket.send(payload)
                        print("Sent transcribed audio")

        except Exception as e:
            print(f"An error occurred: {e}")
            await asyncio.sleep(1)  # Wait before retrying

start_server = websockets.serve(process_input, "0.0.0.0", 8000)
print("Server waiting...")
asyncio.get_event_loop().run_until_complete(start_server)
asyncio.get_event_loop().run_forever()