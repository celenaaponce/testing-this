"""Object detection demo with MobileNet SSD.
This model and code are based on
https://github.com/robmarkcole/object-detection-app
"""

import logging
import queue
from pathlib import Path
from typing import List, NamedTuple

import av
import cv2
import numpy as np
import streamlit as st
from streamlit_webrtc import WebRtcMode, webrtc_streamer
import mediapipe as mp

from turn import get_ice_servers

HERE = Path(__file__).parent
ROOT = HERE.parent

logger = logging.getLogger(__name__)


# MODEL_URL = "https://github.com/robmarkcole/object-detection-app/raw/master/model/MobileNetSSD_deploy.caffemodel"  # noqa: E501
# MODEL_LOCAL_PATH = ROOT / "./models/MobileNetSSD_deploy.caffemodel"
# PROTOTXT_URL = "https://github.com/robmarkcole/object-detection-app/raw/master/model/MobileNetSSD_deploy.prototxt.txt"  # noqa: E501
# PROTOTXT_LOCAL_PATH = ROOT / "./models/MobileNetSSD_deploy.prototxt.txt"

# CLASSES = [
#     "background",
#     "aeroplane",
#     "bicycle",
#     "bird",
#     "boat",
#     "bottle",
#     "bus",
#     "car",
#     "cat",
#     "chair",
#     "cow",
#     "diningtable",
#     "dog",
#     "horse",
#     "motorbike",
#     "person",
#     "pottedplant",
#     "sheep",
#     "sofa",
#     "train",
#     "tvmonitor",
# ]


# class Detection(NamedTuple):
#     class_id: int
#     label: str
#     score: float
#     box: np.ndarray


# @st.cache_resource  # type: ignore
# def generate_label_colors():
#     return np.random.uniform(0, 255, size=(len(CLASSES), 3))


# COLORS = generate_label_colors()

# download_file(MODEL_URL, MODEL_LOCAL_PATH, expected_size=23147564)
# download_file(PROTOTXT_URL, PROTOTXT_LOCAL_PATH, expected_size=29353)


# Session-specific caching
cache_key = "object_detection_dnn"
if cache_key in st.session_state:
    net = st.session_state[cache_key]
# else:
#     net = cv2.dnn.readNetFromCaffe(str(PROTOTXT_LOCAL_PATH), str(MODEL_LOCAL_PATH))
#     st.session_state[cache_key] = net

# score_threshold = st.slider("Score threshold", 0.0, 1.0, 0.5, 0.05)

# NOTE: The callback will be called in another thread,
#       so use a queue here for thread-safety to pass the data
#       from inside to outside the callback.
# TODO: A general-purpose shared state object may be more useful.
result_queue: "queue.Queue[List[Detection]]" = queue.Queue()


def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
    image = frame.to_ndarray(format="bgr24")
    holistic = mp.solutions.holistic.Holistic(
     min_detection_confidence=0.7,
     min_tracking_confidence=0.5
     )
    results = holistic.process(image)
    mp.solutions.drawing_utils.draw_landmarks(image, results.pose_landmarks, mp.solutions.holistic.POSE_CONNECTIONS, 
        mp.solutions.drawing_utils.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4),  
        mp.solutions.drawing_utils.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2) 
        ) 

    # Run inference
    blob = cv2.dnn.blobFromImage(
        cv2.resize(image, (300, 300)), 0.007843, (300, 300), 127.5
    )
    # net.setInput(blob)
    # output = net.forward()

    h, w = image.shape[:2]

    # Convert the output array into a structured form.
    # output = output.squeeze()  # (1, 1, N, 7) -> (N, 7)
    # output = output[output[:, 2] >= score_threshold]
    # detections = [
    #     Detection(
    #         class_id=int(detection[1]),
    #         label=CLASSES[int(detection[1])],
    #         score=float(detection[2]),
    #         box=(detection[3:7] * np.array([w, h, w, h])),
    #     )
    #     for detection in output
    # ]

    # Render bounding boxes and captions
    # for detection in detections:
        # caption = f"{detection.label}: {round(detection.score * 100, 2)}%"
        # color = COLORS[detection.class_id]
        # xmin, ymin, xmax, ymax = detection.box.astype("int")

        # cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color, 2)
        # cv2.putText(
        #     image,
        #     caption,
        #     (xmin, ymin - 15 if ymin - 15 > 15 else ymin + 15),
        #     cv2.FONT_HERSHEY_SIMPLEX,
        #     0.5,
        #     color,
        #     2,
        # )

    # result_queue.put(detections)

    return av.VideoFrame.from_ndarray(image, format="bgr24")


webrtc_ctx = webrtc_streamer(
    key="object-detection",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration={"iceServers": get_ice_servers()},
    video_frame_callback=video_frame_callback,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)


if webrtc_ctx.state.playing:
    labels_placeholder = st.empty()
    # NOTE: The video transformation with object detection and
    # this loop displaying the result labels are running
    # in different threads asynchronously.
    # Then the rendered video frames and the labels displayed here
    # are not strictly synchronized.
    while True:
        result = result_queue.get()
        labels_placeholder.table(result)



# import logging
# import streamlit as st
# from streamlit_webrtc import webrtc_streamer
# import logging
# import queue
# import av
# import streamlit as st
# from streamlit_webrtc import WebRtcMode, webrtc_streamer

# from turn import get_ice_servers
# import cv2
# import streamlit as st
# import mediapipe as mp
# import cv2 as cv
# import numpy as np
# import tempfile
# import time
# from PIL import Image


# # Basic App Scaffolding
# st.title('Face Mesh App using Streamlit')
# result_queue: "queue.Queue[List[Detection]]" = queue.Queue()

# # Resize Images to fit Container
# @st.cache()
# # Get Image Dimensions
# def image_resize(image, width=None, height=None, inter=cv.INTER_AREA):
#     # initialize the dimensions of the image to be resized and
#     dim = None
#     # grab the image size
#     (h,w) = image.shape[:2]

#     if width is None and height is None:
#         return image
#     # calculate the ratio of the height and construct the
#     # dimensions
#     if width is None:
#         r = width/float(w)
#         dim = (int(w*r),height)
#     # calculate the ratio of the width and construct the
#     # dimensions
#     else:
#         r = width/float(w)
#         dim = width, int(h*r)

#     # Resize image
#     resized = cv.resize(image,dim,interpolation=inter)

#     return resized

# def video_frame_callback(frame):
#     image = frame.to_ndarray(format="bgr24")

#     # Run inference
#     blob = cv2.dnn.blobFromImage(
#         cv2.resize(image, (300, 300)), 0.007843, (300, 300), 127.5
#     )
#     net.setInput(blob)
#     output = net.forward()

#     h, w = image.shape[:2]

#     kpil, kpil2, kpil3 = st.columns(3)

#     with kpil:
#         st.markdown('**Word**')
#         kpil_text = st.markdown('0')
    
#     with kpil2:
#         st.markdown('**Correct Signs**')
#         kpil2_text = st.markdown('0')
    
#     dominant_hand = 'LEFT'
#     # keypoint_classifier = KeyPointClassifier()
#     # with open('model/keypoint_classifier/keypoint_classifier_label.csv',
#     #             encoding='utf-8-sig') as f:
#     #     keypoint_classifier_labels = csv.reader(f)
#     #     keypoint_classifier_labels = [
#     #         row[0] for row in keypoint_classifier_labels
#     #     ]
#     pre_processed_landmark_list = None
#     tagged_signs = []
#     success = False
#     after_success = 0
#     hand_sign_id = 1
#     ## Face Mesh
#     holistic = mp.solutions.holistic.Holistic(
#     min_detection_confidence=0.7,
#     min_tracking_confidence=0.5
#     )
        
#     prevTime = 0

#     results = holistic.process(image)
#     mp.solutions.drawing_utils.draw_landmarks(image, results.pose_landmarks, mp.solutions.holistic.POSE_CONNECTIONS, 
#         mp.solutions.drawing_utils.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4),  
#         mp.solutions.drawing_utils.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2) 
#         ) 
#     return av.VideoFrame.from_ndarray(image, format="bgr24")
#     img.flags.writeable = True
#     left_present = dominant_hand == 'LEFT' and results.left_hand_landmarks is not None
#     right_present = dominant_hand == 'RIGHT' and results.right_hand_landmarks is not None
#     face_count = 0
#     if results.pose_landmarks is not None and left_present or right_present and not success:

#         #Face Landmark Drawing
#         for face_landmarks in results.pose_landmarks.landmark:

#             mp.solutions.drawing_utils.draw_landmarks(img, results.pose_landmarks, mp.solutions.holistic.POSE_CONNECTIONS, 
#                     mp.solutions.drawing_utils.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4),  
#                     mp.solutions.drawing_utils.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2) 
#                     ) 
#         if results.left_hand_landmarks:
#         #left eye edge to thumb tip distance
#             x_distance = abs(results.pose_landmarks.landmark[3].x - results.left_hand_landmarks.landmark[4].x)
#             y_distance = abs(results.pose_landmarks.landmark[3].y - results.left_hand_landmarks.landmark[4].y)
#             brect = calc_bounding_rect(img, results.left_hand_landmarks)
#             pre_processed_landmark_list = pre_process_landmark(
#                 results.left_hand_landmarks.landmark)
#             #Face Landmark Drawing
#             for face_landmarks in results.left_hand_landmarks.landmark:

#                 mp.solutions.drawing_utils.draw_landmarks(img, results.left_hand_landmarks, mp.solutions.holistic.HAND_CONNECTIONS, 
#                         mp.solutions.drawing_utils.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4),  
#                         mp.solutions.drawing_utils.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2) 
#                         ) 
#         #     if results.right_hand_landmarks:
#             #right eye edge to thumb tip distance
#         #         x_distance = abs(results.pose_landmarks.landmark[6].x - results.right_hand_landmarks.landmark[4].x)
#         #         y_distance = abs(results.pose_landmarks.landmark[6].y - results.right_hand_landmarks.landmark[4].y)
#         #         brect = calc_bounding_rect(frame, results.right_hand_landmarks)
#         #         pre_processed_landmark_list = pre_process_landmark(
#         #             results.right_hand_landmarks.landmark)
#         #         #Face Landmark Drawing
#         #         for face_landmarks in results.right_hand_landmarks.landmark:

#         #             mp.solutions.drawing_utils.draw_landmarks(frame, results.right_hand_landmarks, mp.solutions.holistic.HAND_CONNECTIONS, 
#         #                     mp.solutions.drawing_utils.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4),  
#         #                     mp.solutions.drawing_utils.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2) 
#         #                     ) 
#             pre_processed_face_landmark_list = pre_process_landmark(results.pose_landmarks.landmark)[:12]
#             total_list = pre_processed_landmark_list + pre_processed_face_landmark_list + [x_distance, y_distance]
#         #     hand_sign_id = keypoint_classifier(total_list)
#         #     tagged_signs.append(hand_sign_id)
#         #     if len(tagged_signs) > 30 and tagged_signs.count(0) > 15 and not success:
#         #         after_success += 1
#         #         success = True

#         # Dashboard
#         # if hand_sign_id == 0:
#         #     output_text = 'Horse 🐴'
#         # else:
#         #     output_text = ''
#         # kpil_text.write(f"<h1 style='text-align: center; color:red;'>{output_text}</h1>", unsafe_allow_html=True)
#         # if success:
#         #     kpil2_text.write(f"<h1 style='text-align: center; color:green;'>Great Job</h1>", unsafe_allow_html=True)
#         #     st.balloons()
#     # img = cv.resize(img,(0,0), fx=0.8, fy=0.8)
#     # img = image_resize(image=img, width=640)
#     return av.VideoFrame.from_ndarray(img, format="bgr24")
#         # stframe.image(frame,channels='BGR', use_column_width=True)


# # Video Page
# st.set_option('deprecation.showfileUploaderEncoding', False)

# use_webcam = True
# ## Get Video
# stframe = st.empty()
# temp_file = tempfile.NamedTemporaryFile(delete=False)
# webrtc_ctx = webrtc_streamer(
#     key="object-detection",
#     # mode=WebRtcMode.SENDRECV,
#     rtc_configuration={"iceServers": get_ice_servers()},
#     video_frame_callback=video_frame_callback,
#     media_stream_constraints={"video": True, "audio": False},
#     async_processing=True,
# )
# if webrtc_ctx.state.playing:
#     labels_placeholder = st.empty()
#     # NOTE: The video transformation with object detection and
#     # this loop displaying the result labels are running
#     # in different threads asynchronously.
#     # Then the rendered video frames and the labels displayed here
#     # are not strictly synchronized.
#     while True:
#         result = result_queue.get()
#         labels_placeholder.table(result)

#     # width = int(webrtc_ctx.get(cv.CAP_PROP_FRAME_WIDTH))
#     # height = int(webrtc_ctx.get(cv.CAP_PROP_FRAME_HEIGHT))
#     # fps_input = int(webrtc_ctx.get(cv.CAP_PROP_FPS))
#     # st.write(width, height, fps_input)

#     # drawing_spec = mp.solutions.drawing_utils.DrawingSpec(thickness=2, circle_radius=1)

#     # kpil, kpil2, kpil3 = st.columns(3)

#     # with kpil:
#     #     st.markdown('**Frame Rate**')
#     #     kpil_text = st.markdown('0')

#     # with kpil2:
#     #     st.markdown('**Detected Faces**')
#     #     kpil2_text = st.markdown('0')

#     # with kpil3:
#     #     st.markdown('**Image Resolution**')
#     #     kpil3_text = st.markdown('0')

#     # st.markdown('<hr/>', unsafe_allow_html=True)


#     # ## Face Mesh
#     # with mp.solutions.face_mesh.FaceMesh(
#     #     max_num_faces=max_faces,
#     #     min_detection_confidence=detection_confidence,
#     #     min_tracking_confidence=tracking_confidence

#     # ) as face_mesh:

#     #         prevTime = 0

#     #         while video.isOpened():
#     #             i +=1
#     #             ret, frame = video.read()
#     #             if not ret:
#     #                 continue

#     #             results = face_mesh.process(frame)
#     #             frame.flags.writeable = True

#     #             face_count = 0
#     #             if results.multi_face_landmarks:

#     #                 #Face Landmark Drawing
#     #                 for face_landmarks in results.multi_face_landmarks:
#     #                     face_count += 1

#     #                     mp.solutions.drawing_utils.draw_landmarks(
#     #                         image=frame,
#     #                         landmark_list=face_landmarks,
#     #                         connections=mp.solutions.face_mesh.FACEMESH_CONTOURS,
#     #                         landmark_drawing_spec=drawing_spec,
#     #                         connection_drawing_spec=drawing_spec
#     #                     )

#     #             # FPS Counter
#     #             currTime = time.time()
#     #             fps = 1/(currTime - prevTime)
#     #             prevTime = currTime

#     #             if record:
#     #                 out.write(frame)

#     #             # Dashboard
#     #             kpil_text.write(f"<h1 style='text-align: center; color:red;'>{int(fps)}</h1>", unsafe_allow_html=True)
#     #             kpil2_text.write(f"<h1 style='text-align: center; color:red;'>{face_count}</h1>", unsafe_allow_html=True)
#     #             kpil3_text.write(f"<h1 style='text-align: center; color:red;'>{width*height}</h1>",
#     #                              unsafe_allow_html=True)

#     #             frame = cv.resize(frame,(0,0), fx=0.8, fy=0.8)
#     #             frame = image_resize(image=frame, width=640)
#     #             stframe.image(frame,channels='BGR', use_column_width=True)
