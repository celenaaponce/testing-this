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
# Video Page



st.set_option('deprecation.showfileUploaderEncoding', False)

drawing_spec = mp.solutions.drawing_utils.DrawingSpec(thickness=2, circle_radius=1)
use_webcam = True
## Get Video
stframe = st.empty()
temp_file = tempfile.NamedTemporaryFile(delete=False)
webrtc_ctx = webrtc_streamer(key="sample", rtc_configuration={"iceServers": get_ice_servers()})    

if webrtc_ctx.video_receiver:

    video_frame = webrtc_ctx.video_receiver.get_frame(timeout=1)
    img_rgb = video_frame.to_ndarray(format="rgb24")
    stframe.image(img_rgb)


    # width = int(video.get(cv.CAP_PROP_FRAME_WIDTH))
    # height = int(video.get(cv.CAP_PROP_FRAME_HEIGHT))
    # fps_input = int(video.get(cv.CAP_PROP_FPS))

    # ## Recording
    # codec = cv.VideoWriter_fourcc('a','v','c','1')
    # out = cv.VideoWriter('output1.mp4', codec, fps_input, (width,height))

    # st.sidebar.text('Input Video')
    # st.sidebar.video(temp_file.name)

    # fps = 0
    # i = 0

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
