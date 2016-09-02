import numpy as np
import matplotlib.pyplot as plt


n_groups = 5

means_men = (84.62, 65.61, 68.57, 69.38, 83.21)
#std_men = (2, 3, 4, 1, 2)

means_women = (80.17, 53.37, 63.62, 50.58, 68.83)
#std_women = (3, 5, 2, 3, 3)

fig, ax = plt.subplots()

index = np.arange(n_groups)
bar_width = 0.35

opacity = 0.4
error_config = {'ecolor': '0.3'}

rects1 = plt.bar(index, means_men, bar_width,
                 alpha=opacity,
                 color='b',
                 error_kw=error_config,
                 label='SVM')

rects2 = plt.bar(index + bar_width, means_women, bar_width,
                 alpha=opacity,
                 color='r',
                 error_kw=error_config,
                 label='KNN')

plt.xlabel('Disease Categories')
plt.ylabel('Accuracy')
plt.title('Accuracy for different disease categories')
plt.xticks(index + bar_width, ('Virus', 'Musculoskeletal', 'Stomatognathic', 'Nervous System', 'Eye'))
plt.legend()

plt.tight_layout()
plt.show()
