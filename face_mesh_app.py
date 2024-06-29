import cv2
import streamlit as st
import mediapipe as mp
import cv2 as cv
import numpy as np
import tempfile
import time
from PIL import Image

DEMO_IMAGE = 'demo/demo.jpg'
DEMO_VIDEO = 'demo/demo.mp4'

# Basic App Scaffolding
st.title('Face Mesh App using Streamlit')

# Resize Images to fit Container
@st.cache()
# Get Image Dimensions
def image_resize(image, width=None, height=None, inter=cv.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    dim = None
    # grab the image size
    (h,w) = image.shape[:2]

    if width is None and height is None:
        return image
    # calculate the ratio of the height and construct the
    # dimensions
    if width is None:
        r = width/float(w)
        dim = (int(w*r),height)
    # calculate the ratio of the width and construct the
    # dimensions
    else:
        r = width/float(w)
        dim = width, int(h*r)

    # Resize image
    resized = cv.resize(image,dim,interpolation=inter)

    return resized

def calc_bounding_rect(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_array = np.empty((0, 2), int)

    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)

        landmark_point = [np.array((landmark_x, landmark_y))]

        landmark_array = np.append(landmark_array, landmark_point, axis=0)

    x, y, w, h = cv.boundingRect(landmark_array)

    return [x, y, x + w, y + h]
def pre_process_landmark(landmark_list):
    x_values = [element.x for element in landmark_list]
    y_values = [element.y for element in landmark_list]

    # temp_landmark_list = copy.deepcopy(landmark_list)
    temp_x = copy.deepcopy(x_values)
    temp_y = copy.deepcopy(y_values)
    # Convert to relative coordinates
    base_x, base_y = 0, 0
    index = 0
    for _ in len(temp_x),:
        if index == 0:
            base_x, base_y = temp_x[index], temp_y[index]         
        temp_x[index] = temp_x[index] - base_x
        temp_y[index] = temp_y[index] - base_y
        index += 1
    # Convert to a one-dimensional list
 
    temp_landmark_list = list(itertools.chain(*zip(temp_x, temp_y))) 

    # Normalization
    max_value = max(list(map(abs, temp_landmark_list)))

    def normalize_(n):
        return n / max_value

    temp_landmark_list = list(map(normalize_, temp_landmark_list))

    return temp_landmark_list

# Video Page


st.set_option('deprecation.showfileUploaderEncoding', False)

use_webcam = True

drawing_spec = mp.solutions.drawing_utils.DrawingSpec(thickness=2, circle_radius=1)

## Get Video
stframe = st.empty()

video = cv.VideoCapture(0)

width = int(video.get(cv.CAP_PROP_FRAME_WIDTH))
height = int(video.get(cv.CAP_PROP_FRAME_HEIGHT))
fps_input = int(video.get(cv.CAP_PROP_FPS))

fps = 0
i = 0

kpil, kpil2, kpil3 = st.columns(3)

with kpil:
    st.markdown('**Word**')
    kpil_text = st.markdown('0')

with kpil2:
    st.markdown('**Correct Signs**')
    kpil2_text = st.markdown('0')

dominant_hand = 'LEFT'
st.markdown('<hr/>', unsafe_allow_html=True)


pre_processed_landmark_list = None
tagged_signs = []
success = False
after_success = 0
hand_sign_id = 1
## Face Mesh
with mp.solutions.holistic.Holistic(
min_detection_confidence=0.7,
min_tracking_confidence=0.5
) as holistic:

        prevTime = 0

        while video.isOpened():
            i +=1
            ret, frame = video.read()
            if not ret:
                continue

            results = holistic.process(frame)
            frame.flags.writeable = True
            left_present = dominant_hand == 'LEFT' and results.left_hand_landmarks is not None
            right_present = dominant_hand == 'RIGHT' and results.right_hand_landmarks is not None
            face_count = 0
            if results.pose_landmarks is not None and left_present or right_present and not success:

                #Face Landmark Drawing
                for face_landmarks in results.pose_landmarks.landmark:

                    mp.solutions.drawing_utils.draw_landmarks(frame, results.pose_landmarks, mp.solutions.holistic.POSE_CONNECTIONS, 
                            mp.solutions.drawing_utils.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4),  
                            mp.solutions.drawing_utils.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2) 
                            )
                    if results.left_hand_landmarks:
                    #left eye edge to thumb tip distance
                        x_distance = abs(results.pose_landmarks.landmark[3].x - results.left_hand_landmarks.landmark[4].x)
                        y_distance = abs(results.pose_landmarks.landmark[3].y - results.left_hand_landmarks.landmark[4].y)
                        brect = calc_bounding_rect(frame, results.left_hand_landmarks)
                        pre_processed_landmark_list = pre_process_landmark(
                            results.left_hand_landmarks.landmark)
                        #Face Landmark Drawing
                        for face_landmarks in results.left_hand_landmarks.landmark:
    
                            solutions.drawing_utils.draw_landmarks(frame, results.left_hand_landmarks, solutions.holistic.HAND_CONNECTIONS, 
                                    solutions.drawing_utils.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4),  
                                    solutions.drawing_utils.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2) 
                                    ) 



            # FPS Counter
            currTime = time.time()
            fps = 1/(currTime - prevTime)
            prevTime = currTime

            # Dashboard
            kpil_text.write(f"<h1 style='text-align: center; color:red;'>{int(fps)}</h1>", unsafe_allow_html=True)
            kpil2_text.write(f"<h1 style='text-align: center; color:red;'>{face_count}</h1>", unsafe_allow_html=True)
            kpil3_text.write(f"<h1 style='text-align: center; color:red;'>{width*height}</h1>",
                             unsafe_allow_html=True)

            frame = cv.resize(frame,(0,0), fx=0.8, fy=0.8)
            frame = image_resize(image=frame, width=640)
            stframe.image(frame,channels='BGR', use_column_width=True)

