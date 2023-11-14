from moviepy.editor import *

clip = (VideoFileClip("./human_traj/test_hirar/test_hirar001.mp4").subclip(1,22))
clip.write_gif("./human_traj/test_hirar/test_hirar001.gif")