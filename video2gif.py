from moviepy.editor import *

clip = (VideoFileClip("./experiments/videos/hierarchical_human_detection/abu_nomask_connectors001_trim.mp4"))
clip.write_gif("./experiments/videos/hierarchical_human_detection/abu_nomask_connectors001_trim.gif",fps=15)