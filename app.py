import streamlit as st
import cv2
from streamlit_webrtc import VideoProcessorBase, webrtc_streamer,WebRtcMode
import numpy as np
st.set_page_config("Face Detector",page_icon="🎥")
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

st.title("Face Detector App")
st.subheader("Detect Faces By uploading Images or using Webcam")

st.sidebar.write("### Your input Source")
opt = st.sidebar.radio("Select Your Option :",["Upload Image","Open video","Real time"])

if opt == "Open video":
    enable = st.checkbox("Open camera")
    video = st.camera_input("Your Camera",disabled=not enable)
    if video:
        file_bytes1 = np.frombuffer(video.read(),np.uint8)
        video = cv2.imdecode(file_bytes1,1)
        gray = cv2.cvtColor(video,cv2.COLOR_BGR2GRAY)
        video = cv2.cvtColor(video,cv2.COLOR_RGB2BGR)

        faces = face_cascade.detectMultiScale(gray,1.1,10)
        for (x,y,w,h) in faces:
            cv2.rectangle(video,(x,y),(x+w,y+h),(0,255,0),3)
        st.image(video,caption=f"Detected {len(faces)} face(s)")

elif opt == "Upload Image":
    img = st.file_uploader("Upload Your Image",type= ["png","jpg","webp","svg"])
    if img:
        file_bytes = np.frombuffer(img.read(), np.uint8)
        img = cv2.imdecode(file_bytes, 1)
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        col1,col2 = st.columns(2)
        with col1:
            st.write("### Your Image : ")
            st.image(img)
        faces = face_cascade.detectMultiScale(gray,1.1,10)
        for (x,y,w,h) in faces:
            cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),7)
        with col2:
            st.write("### After Detecting Faces : ")
            st.image(img,caption=f"face(s) Detected")
else:
    st.subheader("**Real Time Video Capture**")
    class VideoProcessor(VideoProcessorBase):
        def recv(self,frame):
            image = frame.to_ndarray(format = "bgr24")
            gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray,1.1,5)

            for (x,y,w,h) in faces:
                cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0),2)

            return frame.from_ndarray(image,format= "bgr24")
        
    webrtc_streamer(key="Face detector",
                    mode=WebRtcMode.SENDRECV,
                    video_processor_factory=VideoProcessor,
                    rtc_configuration={
                        "iceServers" : [
                            {"urls": ["stun:stun.l.google.com:19302"]},
                        ]
                    },
                    media_stream_constraints={
                        "video":{
                            "frameRate" : {"ideal":40,"max" : 60}
                        },
                        "audio" : False,
                    }
    )