import cv2
from PIL import Image
import logging
import threading
import numpy as np
from keras import models
import torch
import av
import queue
from heart_rate import heartrate
from heart_rate import pyramids
from heart_rate import eulerian
import streamlit as st
from streamlit_webrtc import RTCConfiguration, WebRtcMode, WebRtcStreamerContext, webrtc_streamer
from typing import List, NamedTuple, Optional


freq_min = 1
freq_max = 1.8
logger=logging.getLogger(__name__)
fps=5
RTC_CONFIGURATION=RTCConfiguration(
    {'iceServers':[{'urls':['stun:stun.l.google.com:19302']}]}
)
def main():
    st.header('Web Arayüzü Demo')
    pages={
        'Gerçek zamanlı yüz tanıma':app_object_detection
    }
    pages_titles=pages.keys()
    page_title=st.sidebar.selectbox('Modu seçiniz...',pages_titles)
    st.subheader(page_title)
    page_func=pages[page_title]
    page_func()
    logger.debug('=== Alive threads ===')
    for thread in threading.enumerate():
        if thread.is_alive():
            logger.debug(f'{thread.name} ({thread.ident})')

def app_loopback():
    webrtc_streamer(key='loopback')
    
@st.cache
def detect_face(img):
    detector=torch.hub.load('ultralytics/yolov5','custom', path='models/face.pt')
    mask_model=torch.hub.load('ultralytics/yolov5','custom', path='models/mask.pt')
    gender_model=models.load_model('models/classification_gender_model_utk.h5')
    age_model=models.load_model('models/classification_age_model_utk.h5')
    emotion_model=models.load_model('models/classification_emotion_model_utk.h5')
    mt_res=detector(img)
    return_res=[]
    if len(mt_res.xyxy[0])>0:
        for face in mt_res.xyxy[0]:
            x1,y1,x2,y2,conf,classes=face
            x1,y1,x2,y2,conf,classes=int(x1),int(y1),int(x2),int(y2),float(conf),float(classes)            
            width=x2-x1
            height=y2-y1 
            center=[x1+(width/2),y1+(height/2)]
            max_border=max(width,height)

            left=max(int(center[0]-(max_border/2)),0)
            right=max(int(center[0]+(max_border/2)),0)
            top=max(int(center[1]-(max_border/2)),0)
            bottom=max(int(center[1]+(max_border/2)),0)

            center_img_k = img[top:top+max_border, left:left+max_border, :]
            center_img_m = img[top:top+max_border+30, left:left+max_border+30, :]
            #center_img_m=cv2.cvtColor(center_img_m,cv2.COLOR_BGR2RGB)
            center_img=np.array(Image.fromarray(center_img_k).resize([224,224]))
            gender_preds=gender_model.predict(center_img.reshape(1,224,224,3))[0][0]
            age_preds=age_model.predict(center_img.reshape(1,224,224,3))[0][0]

            resized_img=np.array(Image.fromarray(center_img_k).resize([64,64]))
            emotion_preds=emotion_model.predict(resized_img.reshape(1,64,64,3))
            mask_pred=mask_model(img)
            mask_coords=mask_pred.xyxy

            roi_frame = cv2.resize(center_img_k, (500, 500))
                
            frame = np.ndarray(shape=roi_frame.shape, dtype="float")
            
            frame[:] = roi_frame * (1. / 255)
            lap_video=pyramids.build_video_pyramid(frame)
            
            for i, video in enumerate(lap_video):
                if i == 0 or i == len(lap_video)-1:
                    continue
                result, fft, frequencies = eulerian.fft_filter(video, freq_min, freq_max, fps)
                lap_video[i] += result
                print("Calculating heart rate...")
                heart_rate = heartrate.find_heart_rate(fft, frequencies, freq_min, freq_max)
            return_res.append([top,right,bottom,left,gender_preds,age_preds,emotion_preds,mask_coords,heart_rate])
    return return_res

def app_object_detection():
    CLASSES=[
        'face'
    ]
    @st.experimental_singleton
    def generate_label_colors():
            return np.random.uniform(0,255, size=(len(CLASSES),3))
    COLORS=generate_label_colors()
    class Detection(NamedTuple):
        name:str
        age:float
        gender:str
        emotion:str
        mask:str
        heart_rate:float
    def _annotate_image(frame,detections):
        emotion_dict={
            0: 'Anger',
            1: 'Contemt',
            2: 'Disgust',
            3: 'Fear',
            4: 'Happy',
            5: 'Saddness',
            6: 'Surprise'
        }
        mask_dict={
            0: 'Mask',
            1: 'No-Mask',
            2: 'None'
        }
        result:List[Detection]=[]
        for top,right,bottom,left,gender_preds,age_preds,emotion_preds,mask_coords,heart_rate in detections:
            cv2.rectangle(frame,(left,top),(right,bottom),(255,0,0),2)
            gender_text='Female' if gender_preds>0.5 else 'Male'
            cv2.putText(frame,'Gender: {}({:.3f})'.format(gender_text,gender_preds),(left,top-10),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,93,255),1,cv2.LINE_AA)
            cv2.putText(frame,'Age: {:.3f}'.format(age_preds),(left,top-25),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,93,2552),1,cv2.LINE_AA)
            cv2.putText(frame,'Emotion: {}({:.3f})'.format(emotion_dict[np.argmax(emotion_preds)],np.max(emotion_preds)),(left,top-40),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,93,255),1,cv2.LINE_AA)
            classes=2
            if len(mask_coords[0])>0:
                xm1,ym1,xm2,ym2,conf,classes=mask_coords[0][0]
                cv2.putText(frame,'{}: {:.2f}'.format(mask_dict[int(classes)],conf),(left,top+15),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,93,255),1,cv2.LINE_AA)
            
            cv2.putText(frame,'Heart Rate: {:.2f} BPM'.format(heart_rate),(left,top+30),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,93,255),1,cv2.LINE_AA)
            result.append(Detection(name='face',age=age_preds,gender=gender_text,emotion=emotion_dict[np.argmax(emotion_preds)],mask=mask_dict[int(classes)],heart_rate=heart_rate))
        return frame,result
    result_queue=(queue.Queue())

    def callback(frame:av.VideoFrame)->av.VideoFrame:
        image=frame.to_ndarray(format='bgr24')
        face_location=detect_face(image)
        annotated_frame,result=_annotate_image(image,face_location)
        result_queue.put(result)
        return av.VideoFrame.from_ndarray(annotated_frame,format='bgr24')
    webrtc_ctx=webrtc_streamer(
        key='object_detection',
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTC_CONFIGURATION,
        video_frame_callback=callback,
        media_stream_constraints={'video':True,'audio':False},
        async_processing=True
    )
    if st.checkbox('Algılanan yüzleri göster',value=True):
        if webrtc_ctx.state.playing:
            labels_placeholder=st.empty()
            while True:
                try:
                    result=result_queue.get(timeout=1.0)
                except queue.Empty:
                    result=None
                labels_placeholder.table(result)
if __name__ == '__main__':
    import os
    DEBUG = os.environ.get('DEBUG', 'false').lower() not in [
        'false', 'no', '0']
    logging.basicConfig(format='[%(asctime)s] %(levelname)7s from %(name)s in %(pathname)s:%(lineno)d: '
                        '%(message)s',
                        force=True)
    logger.setLevel(level=logging.DEBUG if DEBUG else logging.INFO)
    st_webrtc_logger = logging.getLogger('streamlit_webrtc')
    st_webrtc_logger.setLevel(logging.DEBUG)
    fsevents_logger = logging.getLogger('fsevents')
    fsevents_logger.setLevel(logging.WARNING)
    main()
