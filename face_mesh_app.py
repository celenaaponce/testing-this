import logging
import queue
from pathlib import Path
from typing import List, NamedTuple
from turn import get_ice_servers
import av
import cv2
import gc
import numpy as np
import streamlit as st
from streamlit_webrtc import WebRtcMode, webrtc_streamer

import mediapipe as mp

logger = logging.getLogger(__name__)

# Initialize Mediapipe Holistic
mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic()
pre_processed_landmark_list = None
tagged_signs = []
success = False
after_success = 0
hand_sign_id = 1
dominant_hand = 'LEFT'
# Define a NamedTuple for the holistic results if needed
class HolisticResult(NamedTuple):
    landmarks: np.ndarray
    # Add other attributes as needed

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

# Session-specific caching
cache_key = "mediapipe_holistic"
if cache_key not in st.session_state:
    st.session_state[cache_key] = holistic
# Reduce frame resolution
TARGET_WIDTH = 640
TARGET_HEIGHT = 480

# Limit the queue size
QUEUE_MAX_SIZE = 10
result_queue: "queue.Queue[List[HolisticResult]]" = queue.Queue(maxsize=QUEUE_MAX_SIZE)

def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:

    image = frame.to_ndarray(format="bgr24")
    image = cv2.resize(image, (TARGET_WIDTH, TARGET_HEIGHT))
    
    # Convert the BGR image to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Process the image and detect holistic landmarks
    results = holistic.process(image_rgb)
    # left_present = dominant_hand == 'LEFT' and results.left_hand_landmarks is not None
    # right_present = dominant_hand == 'RIGHT' and results.right_hand_landmarks is not None
    # if results.left_hand_landmarks:
    # #left eye edge to thumb tip distance
    #     x_distance = abs(results.pose_landmarks.landmark[3].x - results.left_hand_landmarks.landmark[4].x)
    #     y_distance = abs(results.pose_landmarks.landmark[3].y - results.left_hand_landmarks.landmark[4].y)
    #     brect = calc_bounding_rect(frame, results.left_hand_landmarks)
    #     pre_processed_landmark_list = pre_process_landmark(
    #         results.left_hand_landmarks.landmark)  
    # Draw landmarks on the image
    mp.solutions.drawing_utils.draw_landmarks(
        image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
    mp.solutions.drawing_utils.draw_landmarks(
        image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    mp.solutions.drawing_utils.draw_landmarks(
        image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    # pre_processed_face_landmark_list = pre_process_landmark(results.pose_landmarks.landmark)[:12]
    # total_list = pre_processed_landmark_list + pre_processed_face_landmark_list + [x_distance, y_distance]  
    # Extract landmarks and other data if needed
    landmarks = results.pose_landmarks.landmark if results.pose_landmarks else []
    result_queue.put(landmarks)
    # Clear Mediapipe results to free memory
    del results
    gc.collect()  # Trigger garbage collection
    return av.VideoFrame.from_ndarray(image, format="bgr24")

webrtc_ctx = webrtc_streamer(
    key="holistic-detection",
    mode=WebRtcMode.SENDRECV,
    video_frame_callback=video_frame_callback,
    rtc_configuration={"iceServers": get_ice_servers()},
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)

# if st.checkbox("Show the detected landmarks", value=True):
if webrtc_ctx.state.playing:
    landmarks_placeholder = st.empty()
    while True:
        result = result_queue.get()
            # landmarks_placeholder.table(result)




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
