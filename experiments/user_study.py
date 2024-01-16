import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
FILE_DIR = Path(__file__).resolve().parent
import os


vision_time = [1,1]
vision_expectation = [0.6,0.58]
speech_time = [0.97,0.78,1]
speech_expectation = [0.85,0.58,0.9]
multimodal_time = [0.89,0.67,0.88]
multimodal_expectation = [1,0.88,0.96]


# plt.plot(vision_expectation,vision_time,label="vision")
# plt.plot(speech_expectation,speech_time,label="speech")
# plt.plot(multimodal_expectation,multimodal_time,label="multimodal")

task1 = [(0.6,1),(0.85,0.97),(1,0.89)]
task2 = [(0.58,1),(0.58,0.78),(0.88,0.67)]
task3 = [(0.9,1),(0.96,0.88)]
plt.plot(task1,vision_time,label="vision")
plt.plot(speech_expectation,speech_time,label="speech")
plt.plot(multimodal_expectation,multimodal_time,label="multimodal")
plt.legend()
plt.savefig(f'{FILE_DIR}/user_study.png')