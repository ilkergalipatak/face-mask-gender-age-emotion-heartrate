import cv2
import pyramids
import heartrate
import preprocessing
import eulerian

freq_min=1
freq_max=1.0

print('Reading + processing')
video_frames,frame_ct,fps=preprocessing.read_video()

print('Building Laplacian video pyramid')
lap_video=pyramids.build_video_pyramid(video_frames)

amplified_video_pyramid=[]
heart_rate=0.0
for i,video in enumerate(lap_video):
    if i==0 or i==len(lap_video)-1:
        continue
    print("Running FFT and Eulerian magnification...")
    result,fft,frquencies=eulerian.fft_filter(video,freq_min,freq_max,fps)
    lap_video[i]+=result
    print("Calculating heart rate...")
    heart_rate=heartrate.find_heart_rate(fft,frquencies,freq_min,freq_max)
print("Rebuilding final video...")
amplified_frames=pyramids.collapse_laplacian_video_pyramid(lap_video,frame_ct)

print('Heart rate: ',heart_rate, ' bpm')

for frame in amplified_frames:
    cv2.imshow('frame',frame)
    cv2.waitKey(20)