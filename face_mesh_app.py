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

if True:

    st.set_option('deprecation.showfileUploaderEncoding', False)

    use_webcam = st.sidebar.button('Use Webcam')

    drawing_spec = mp.solutions.drawing_utils.DrawingSpec(thickness=2, circle_radius=1)

    ## Get Video
    stframe = st.empty()
    # video_file_buffer = st.sidebar.file_uploader("Upload a Video", type=['mp4', 'mov', 'avi', 'asf', 'm4v'])
    # temp_file = tempfile.NamedTemporaryFile(delete=False)
    img_file_buffer = st.camera_input("Take a picture")

    if img_file_buffer is not None:
    # To read image file buffer with OpenCV:
        bytes_data = img_file_buffer.getvalue()
        cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)


    drawing_spec = mp.solutions.drawing_utils.DrawingSpec(thickness=2, circle_radius=1)

    kpil, kpil2, kpil3 = st.columns(3)

    with kpil:
        st.markdown('**Frame Rate**')
        kpil_text = st.markdown('0')

    with kpil2:
        st.markdown('**Detected Faces**')
        kpil2_text = st.markdown('0')

    with kpil3:
        st.markdown('**Image Resolution**')
        kpil3_text = st.markdown('0')

    st.markdown('<hr/>', unsafe_allow_html=True)


    ## Face Mesh
    with solutions.holistic.Holistic(
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
    ) as holistic:

            prevTime = 0

            while img_file_buffer is not None:
                i +=1
                ret, frame = cv2_img.read()
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
    
                        solutions.drawing_utils.draw_landmarks(frame, results.pose_landmarks, solutions.holistic.POSE_CONNECTIONS, 
                                solutions.drawing_utils.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4),  
                                solutions.drawing_utils.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2) 
                                ) 


                # FPS Counter
                currTime = time.time()
                fps = 1/(currTime - prevTime)
                prevTime = currTime

                if record:
                    out.write(frame)

                # Dashboard
                kpil_text.write(f"<h1 style='text-align: center; color:red;'>{int(fps)}</h1>", unsafe_allow_html=True)
                kpil2_text.write(f"<h1 style='text-align: center; color:red;'>{face_count}</h1>", unsafe_allow_html=True)
                kpil3_text.write(f"<h1 style='text-align: center; color:red;'>{width*height}</h1>",
                                 unsafe_allow_html=True)

                frame = cv.resize(frame,(0,0), fx=0.8, fy=0.8)
                frame = image_resize(image=frame, width=640)
                stframe.image(frame,channels='BGR', use_column_width=True)

