import cv2
import numpy as np
import pyramids
import eulerian
import heartrate
face_cascade=cv2.CascadeClassifier('frontalface.xml')

# def read_video():
cap = cv2.VideoCapture(0)
fps = int(cap.get(cv2.CAP_PROP_FPS))
#     fps=int(cap.get(cv2.CAP_PROP_FPS))
#     video_frames=[]
#     face_rects=()
freq_min = 1
freq_max = 1.8
while True:
    ret,img=cap.read()
    if not ret:
        break
    
    gray_img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    
    face_rects=face_cascade.detectMultiScale(gray_img,1.3,5)

    if len(face_rects)>0:
        for (x,y,w,h) in face_rects:
            roi_frame=img[y:y+h,x:x+w]
            cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
            if roi_frame.size != img.size:
                roi_frame = cv2.resize(roi_frame, (500, 500))
                
                frame = np.ndarray(shape=roi_frame.shape, dtype="float")
                
                frame[:] = roi_frame * (1. / 255)
                lap_video=pyramids.build_video_pyramid(frame)
                
                for i, video in enumerate(lap_video):
                    if i == 0 or i == len(lap_video)-1:
                        continue
                    print("Running FFT and Eulerian magnification...")
                    result, fft, frequencies = eulerian.fft_filter(video, freq_min, freq_max, fps)
                    lap_video[i] += result

                    print("Calculating heart rate...")
                    heart_rate = heartrate.find_heart_rate(fft, frequencies, freq_min, freq_max)
                    print(heart_rate)

    cv2.imshow('Video',img)
    if cv2.waitKey(1)==113:
        break
cap.release()
cv2.destroyAllWindows()
# return video_frames,frame_ct,fps