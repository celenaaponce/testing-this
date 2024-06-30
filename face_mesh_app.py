import logging
import streamlit as st
from streamlit_webrtc import webrtc_streamer
import logging
import queue

import streamlit as st
from streamlit_webrtc import WebRtcMode, webrtc_streamer

from turn import get_ice_servers
import cv2
import streamlit as st
import mediapipe as mp
import cv2 as cv
import numpy as np
import tempfile
import time
from PIL import Image


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

def video_frame_callback(frame):
    img = frame.to_ndarray(format="bgr24")

    flipped = img[::-1,:,:]

    return av.VideoFrame.from_ndarray(flipped, format="bgr24")
    kpil, kpil2, kpil3 = st.columns(3)

    with kpil:
        st.markdown('**Word**')
        kpil_text = st.markdown('0')
    
    with kpil2:
        st.markdown('**Correct Signs**')
        kpil2_text = st.markdown('0')
    
    dominant_hand = 'LEFT'
    st.markdown('<hr/>', unsafe_allow_html=True)
    # keypoint_classifier = KeyPointClassifier()
    # with open('model/keypoint_classifier/keypoint_classifier_label.csv',
    #             encoding='utf-8-sig') as f:
    #     keypoint_classifier_labels = csv.reader(f)
    #     keypoint_classifier_labels = [
    #         row[0] for row in keypoint_classifier_labels
    #     ]
    pre_processed_landmark_list = None
    tagged_signs = []
    success = False
    after_success = 0
    hand_sign_id = 1
    ## Face Mesh
    with solutions.holistic.Holistic(
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
    ) as holistic:
        
        prevTime = 0

        results = holistic.process(frame)
        return results
        frame.flags.writeable = True
        left_present = dominant_hand == 'LEFT' and results.left_hand_landmarks is not None
        right_present = dominant_hand == 'RIGHT' and results.right_hand_landmarks is not None
        face_count = 0
        st.write(results)
        if results.pose_landmarks is not None and left_present or right_present and not success:

            #Face Landmark Drawing
            for face_landmarks in results.pose_landmarks.landmark:

                solutions.drawing_utils.draw_landmarks(frame, results.pose_landmarks, solutions.holistic.POSE_CONNECTIONS, 
                        solutions.drawing_utils.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4),  
                        solutions.drawing_utils.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2) 
                        ) 
        #     if results.left_hand_landmarks:
        #     #left eye edge to thumb tip distance
        #         x_distance = abs(results.pose_landmarks.landmark[3].x - results.left_hand_landmarks.landmark[4].x)
        #         y_distance = abs(results.pose_landmarks.landmark[3].y - results.left_hand_landmarks.landmark[4].y)
        #         brect = calc_bounding_rect(frame, results.left_hand_landmarks)
        #         pre_processed_landmark_list = pre_process_landmark(
        #             results.left_hand_landmarks.landmark)
        #         #Face Landmark Drawing
        #         for face_landmarks in results.left_hand_landmarks.landmark:

        #             solutions.drawing_utils.draw_landmarks(frame, results.left_hand_landmarks, solutions.holistic.HAND_CONNECTIONS, 
        #                     solutions.drawing_utils.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4),  
        #                     solutions.drawing_utils.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2) 
        #                     ) 
        # #     if results.right_hand_landmarks:
        # #     #right eye edge to thumb tip distance
        # #         x_distance = abs(results.pose_landmarks.landmark[6].x - results.right_hand_landmarks.landmark[4].x)
        # #         y_distance = abs(results.pose_landmarks.landmark[6].y - results.right_hand_landmarks.landmark[4].y)
        # #         brect = calc_bounding_rect(frame, results.right_hand_landmarks)
        # #         pre_processed_landmark_list = pre_process_landmark(
        # #             results.right_hand_landmarks.landmark)
        # #         #Face Landmark Drawing
        # #         for face_landmarks in results.right_hand_landmarks.landmark:

        # #             mp.solutions.drawing_utils.draw_landmarks(frame, results.right_hand_landmarks, mp.solutions.holistic.HAND_CONNECTIONS, 
        # #                     mp.solutions.drawing_utils.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4),  
        # #                     mp.solutions.drawing_utils.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2) 
        # #                     ) 
        #     pre_processed_face_landmark_list = pre_process_landmark(
        #     results.pose_landmarks.landmark)[:12]
        #     total_list = pre_processed_landmark_list + pre_processed_face_landmark_list + [x_distance, y_distance]
        #     hand_sign_id = keypoint_classifier(total_list)
        #     tagged_signs.append(hand_sign_id)
        #     if len(tagged_signs) > 30 and tagged_signs.count(0) > 15 and not success:
        #         after_success += 1
        #         success = True

        # # Dashboard
        # if hand_sign_id == 0:
        #     output_text = 'Horse 🐴'
        # else:
        #     output_text = ''
        # kpil_text.write(f"<h1 style='text-align: center; color:red;'>{output_text}</h1>", unsafe_allow_html=True)
        # if success:
        #     kpil2_text.write(f"<h1 style='text-align: center; color:green;'>Great Job</h1>", unsafe_allow_html=True)
        #     st.balloons()
        # frame = cv.resize(frame,(0,0), fx=0.8, fy=0.8)
        # frame = image_resize(image=frame, width=640)
        # stframe.image(frame,channels='BGR', use_column_width=True)


# Video Page
st.set_option('deprecation.showfileUploaderEncoding', False)

use_webcam = True
## Get Video
stframe = st.empty()
temp_file = tempfile.NamedTemporaryFile(delete=False)
webrtc_streamer(key="sample", rtc_configuration={"iceServers": get_ice_servers()}, video_frame_callback=video_frame_callback)    


    # width = int(webrtc_ctx.get(cv.CAP_PROP_FRAME_WIDTH))
    # height = int(webrtc_ctx.get(cv.CAP_PROP_FRAME_HEIGHT))
    # fps_input = int(webrtc_ctx.get(cv.CAP_PROP_FPS))
    # st.write(width, height, fps_input)

    # drawing_spec = mp.solutions.drawing_utils.DrawingSpec(thickness=2, circle_radius=1)

    # kpil, kpil2, kpil3 = st.columns(3)

    # with kpil:
    #     st.markdown('**Frame Rate**')
    #     kpil_text = st.markdown('0')

    # with kpil2:
    #     st.markdown('**Detected Faces**')
    #     kpil2_text = st.markdown('0')

    # with kpil3:
    #     st.markdown('**Image Resolution**')
    #     kpil3_text = st.markdown('0')

    # st.markdown('<hr/>', unsafe_allow_html=True)


    # ## Face Mesh
    # with mp.solutions.face_mesh.FaceMesh(
    #     max_num_faces=max_faces,
    #     min_detection_confidence=detection_confidence,
    #     min_tracking_confidence=tracking_confidence

    # ) as face_mesh:

    #         prevTime = 0

    #         while video.isOpened():
    #             i +=1
    #             ret, frame = video.read()
    #             if not ret:
    #                 continue

    #             results = face_mesh.process(frame)
    #             frame.flags.writeable = True

    #             face_count = 0
    #             if results.multi_face_landmarks:

    #                 #Face Landmark Drawing
    #                 for face_landmarks in results.multi_face_landmarks:
    #                     face_count += 1

    #                     mp.solutions.drawing_utils.draw_landmarks(
    #                         image=frame,
    #                         landmark_list=face_landmarks,
    #                         connections=mp.solutions.face_mesh.FACEMESH_CONTOURS,
    #                         landmark_drawing_spec=drawing_spec,
    #                         connection_drawing_spec=drawing_spec
    #                     )

    #             # FPS Counter
    #             currTime = time.time()
    #             fps = 1/(currTime - prevTime)
    #             prevTime = currTime

    #             if record:
    #                 out.write(frame)

    #             # Dashboard
    #             kpil_text.write(f"<h1 style='text-align: center; color:red;'>{int(fps)}</h1>", unsafe_allow_html=True)
    #             kpil2_text.write(f"<h1 style='text-align: center; color:red;'>{face_count}</h1>", unsafe_allow_html=True)
    #             kpil3_text.write(f"<h1 style='text-align: center; color:red;'>{width*height}</h1>",
    #                              unsafe_allow_html=True)

    #             frame = cv.resize(frame,(0,0), fx=0.8, fy=0.8)
    #             frame = image_resize(image=frame, width=640)
    #             stframe.image(frame,channels='BGR', use_column_width=True)
