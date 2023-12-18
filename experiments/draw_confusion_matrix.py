import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn import metrics
import numpy as np
from pathlib import Path
FILE_DIR = Path(__file__).resolve().parent
import os

os.environ["DISPLAY"]=":0"

# if __name__ == "__main__":
    # restrict = "no"
    # if restrict == "no":

    # intention_list = np.array(intention_list)
    # labels_list = np.array(labels_list)
    
    # confusion_matrix = []
    # if restrict == "no":
    #     confusion_matrix.append([13,13,3,1])
    #     confusion_matrix.append([1,29,0,0])
    #     confusion_matrix.append([2,0,27,1])
    #     confusion_matrix.append([0,0,3,27])
    # elif restrict == "ood":
    #     confusion_matrix.append([21,6,2,1])
    #     confusion_matrix.append([1,29,0,0])
    #     confusion_matrix.append([4,0,26,0])
    #     confusion_matrix.append([3,0,1,26])
    # elif restrict == "all":
    #     confusion_matrix.append([30,0,0,0])
    #     confusion_matrix.append([2,28,0,0])
    #     confusion_matrix.append([4,0,26,0])
    #     confusion_matrix.append([5,0,0,25])
    # cm_display_norm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = ["no_action","connectors","screws","wheels"])
    # cm_display_norm_display.plot(ax=plt.gca()) 
    # if restrict == "no":
    #     plt.text(1, 0, confusion_matrix[0,1], color='red', ha='center', va='center')
    #     plt.text(2, 0, confusion_matrix[0,2], color='red', ha='center', va='center')
    #     plt.text(3, 0, confusion_matrix[0,3], color='red', ha='center', va='center')
    #     plt.text(2, 3, confusion_matrix[3,2], color='red', ha='center', va='center')
    #     plt.text(3, 2, confusion_matrix[2,3], color='red', ha='center', va='center')
    # if restrict == "ood":

    #     plt.text(1, 0, confusion_matrix[0,1], color='green', ha='center', va='center')
    #     plt.text(2, 0, confusion_matrix[0,2], color='green', ha='center', va='center')
    #     plt.text(3, 0, confusion_matrix[0,3], color='green', ha='center', va='center')
    #     plt.text(2, 3, confusion_matrix[3,2], color='green', ha='center', va='center')
    #     plt.text(3, 2, confusion_matrix[2,3], color='green', ha='center', va='center')

    # plt.savefig(f'{FILE_DIR}/../confusion_matrix/cm_{restrict}.jpg', bbox_inches = 'tight')

# category_names = ['Confidence+Working Area Restriction','Confidence Restriction','No Restriction']
# results = {
#     'Error 1': [0, 0.30, 0.57],
#     'Error 2': [0, 0.01, 0.04],
#     'Error 3': [0.12, 0.08, 0.03]
# }

# labels = ['Error 1', 'Error 2', 'Error 3']
# all_means = np.array([0,0,0.08])
# conf_means = np.array([0.30,0.01,0.05])
# no_means = np.array([0.57,0.04,0.03])
# all_std = np.array([0,0,0.02])
# conf_std = np.array([0.04,0.005,0.015])
# no_std = np.array([0.05,0.02,0.01])
# width = 0.50      # the width of the bars: can also be len(x) sequence

# fig, ax = plt.subplots()

# ax.bar(labels, no_means, width, yerr=no_std, 
#        label='No Restrictionn')
# ax.bar(labels, conf_means, width, yerr=conf_std, bottom=no_means,
#        label='Confidence Restriction')
# ax.bar(labels, all_means, width, yerr=all_std, bottom=conf_means+no_means,
#        label='Confidence+Working Area Restriction')

labels = ['No', 'Confidence', 'Confidence+Working Area']
error1_means = np.array([0.57,0.30,0])
error2_means = np.array([0.04,0.01,0])
error3_means = np.array([0.03,0.08,0.12])
error1_std = np.array([0.02,0.01,0])
error2_std = np.array([0.01,0.005,0])
error3_std = np.array([0.005,0.01,0.01])
width = 0.50      # the width of the bars: can also be len(x) sequence

fig, ax = plt.subplots()

ax.bar(labels, error1_means, width, yerr=error1_std, 
       label='Mis-predicting "no action"')
ax.bar(labels, error2_means, width, yerr=error2_std, bottom=error1_means,
       label='Predicting one intention as another one')
ax.bar(labels, error3_means, width, yerr=error3_std, bottom=error1_means+error2_means,
       label='Predicting intentions as "no action"')

ax.set_ylabel('Error Rate')
ax.set_title('Error Rate of Different Restrictions')
ax.legend()

plt.savefig(f'{FILE_DIR}/restriction/error_rate_diff_restrictions.jpg', bbox_inches = 'tight')