#Convert video mp4 to other format
$VIDEO="C:\Programming\MPQUIC-meta-learning\Code\main\video\BigBuckBunny.mp4"

cd conv_video

#ffmpeg -y -i $VIDEO -c:a aac -ac 2 -ab 128k -c:v libx264 -force_key_frames 'expr:gte(t,n_forced*2)' -b:v 250k -maxrate 500k -bufsize 1000k -s 426x240 -vf scale=426x240 -r 24 video_240_25fps.mp4
#ffmpeg -y -i $VIDEO -c:a aac -ac 2 -ab 128k -c:v libx264 -force_key_frames 'expr:gte(t,n_forced*2)' -b:v 400k -maxrate 800k -bufsize 1600k -s 640x360 -vf scale=640x360 -r 24 video_360_25fps.mp4

#ffmpeg -r 25 -y -i $VIDEO -c:a aac -ac 2 -ab 128k -c:v libx264 -force_key_frames 'expr:gte(t,n_forced*2)' -x264-params min-keyint=49:keyint=49:keyint_min=49:scenecut=0 -b:v 250k -maxrate 500k -bufsize 1000k -s 426x240 -vf scale=426x240 -r 24 video_240_25fps.mp4
#ffmpeg -r 25 -y -i $VIDEO -c:a aac -ac 2 -ab 128k -c:v libx264 -keyint_min 48 -x264-params keyint=48:keyint-min=48:min-keyint=48:scenecut=0 -b:v 250k -maxrate 500k -bufsize 1000k -s 426x240 -vf scale=426x240 -r 24 video_240_25fps.mp4
#ffmpeg -y -i $VIDEO -c:a aac -ac 2 -ab 128k -c:v libx264 -force_key_frames 'expr:gte(t,n_forced*2)' -keyint_min 96 -x264-params keyint=25:keyint-min=96:min-keyint=96:scenecut=0 -b:v 250k -maxrate 500k -bufsize 1000k -s 426x240 -vf scale=426x240 -24 video_240_25fps.mp4

ffmpeg -y -i $VIDEO -c:a aac -ac 2 -ab 128k -c:v libx264 -x264opts keyint=96:min-keyint=96:no-scenecut -b:v 250k -maxrate 500k -bufsize 1000k -s 426x240 -vf scale=426x240 -filter:v fps=25 video_240_25fps.mp4
ffmpeg -y -i $VIDEO -c:a aac -ac 2 -ab 128k -c:v libx264 -x264opts keyint=96:min-keyint=96:no-scenecut -b:v 400k -maxrate 800k -bufsize 1600k -s 640x360 -vf scale=640x360 -filter:v fps=25 video_360_25fps.mp4
ffmpeg -y -i $VIDEO -c:a aac -ac 2 -ab 128k -c:v libx264 -x264opts keyint=96:min-keyint=96:no-scenecut -b:v 500k -maxrate 1000k -bufsize 2000k -s 854x480 -vf scale=854:480 -filter:v fps=25 video_480_25fps.mp4
ffmpeg -y -i $VIDEO -c:a aac -ac 2 -ab 128k -c:v libx264 -x264opts keyint=96:min-keyint=96:no-scenecut -b:v 1500k -maxrate 3000k -bufsize 6000k -s 1280x720 -vf scale=1280:720 -filter:v fps=25 video_720_25fps.mp4
ffmpeg -y -i $VIDEO -c:a aac -ac 2 -ab 128k -c:v libx264 -x264opts keyint=96:min-keyint=96:no-scenecut -b:v 2250k -maxrate 4500k -bufsize 9000k -s 1280x720 -vf scale=1280:720 -filter:v fps=60 video_720_60fps.mp4
ffmpeg -y -i $VIDEO -c:a aac -ac 2 -ab 128k -c:v libx264 -x264opts keyint=96:min-keyint=96:no-scenecut -b:v 3000k -maxrate 6000k -bufsize 12000k -s 1920x1080 -vf scale=1920:1080 -filter:v fps=25 video_1080_25fps.mp4
ffmpeg -y -i $VIDEO -c:a aac -ac 2 -ab 128k -c:v libx264 -x264opts keyint=96:min-keyint=96:no-scenecut -b:v 4500k -maxrate 9000k -bufsize 18000k -s 1920x1080 -vf scale=1920:1080 -filter:v fps=60 video_1080_60fps.mp4


# Split the video and create the representation in mpd
# the number 4000 and 4 have to change to the value for the second per desired segment.
# For example, if you want segments with 1 second video, you need to put 1000 and 1
MP4Box -dash 4000 `
-segment-name 'segment_$RepresentationID$_' `
-mpd-refresh 4 `
-fps 25 video_240_25fps.mp4`#video:id=240p `
-fps 25 video_360_25fps.mp4`#video:id=360p `
-fps 25 video_480_25fps.mp4`#video:id=480p `
-fps 25 video_720_25fps.mp4`#video:id=720p `
-fps 60 video_720_60fps.mp4`#video:id=7202p `
-fps 25 video_1080_25fps.mp4`#video:id=1080p `
-fps 60 video_1080_60fps.mp4`#video:id=10802p `
-out dash\output_dash.mpd
#-fps 30 video_1440_30fps.mp4#video:id=1440p \
#-fps 60 video_1440_60fps.mp4#video:id=14402p \
#-fps 30 video_2160_30fps.mp4#video:id=2560p \
#-fps 60 video_2160_60fps.mp4#video:id=25602p \
#-rap `
cd ..
#Ps: For the SARA algorithm, use the python code "modify_mpd" to convert the mpd representative to be compatible with SARA

