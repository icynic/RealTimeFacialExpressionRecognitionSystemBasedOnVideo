import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os

plt.rcParams['font.family'] = 'SimHei'

fig, axs = plt.subplots(nrows=4, ncols=8, figsize=(16, 8))
dataset_names = ['FER-2013', 'AffectNet', 'JAFFE', 'CK+']
dataset_n= ['fer', 'aff', 'ja', 'ck']
emotion_labels = ['愤怒', '蔑视', '厌恶', '恐惧', '快乐', '中性', '悲伤', '惊讶']

for i, dataset in enumerate(dataset_names):
    for j in range(8):
        img_path = f'C:/Users/effax/Desktop/{dataset_n[i]}{j+1}.jpg'  # 假设你已经准备好图片
        if os.path.exists(img_path):
            img = mpimg.imread(img_path)
            if i !=1:
                axs[i, j].imshow(img, cmap='gray')
            else:
                axs[i, j].imshow(img)
        axs[i, j].tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)  # 不显示刻度和刻度文字
        axs[i, j].spines['top'].set_visible(False)
        axs[i, j].spines['right'].set_visible(False)
        axs[i, j].spines['bottom'].set_visible(False)
        axs[i, j].spines['left'].set_visible(False)

        if i == 0:
            axs[i, j].set_title(emotion_labels[j], fontsize=30)
        if j == 0:
            axs[i, j].set_ylabel(dataset_names[i], fontsize=25, labelpad=60, rotation=0)

plt.tight_layout()
plt.savefig('emotion_dataset_grid.png')
