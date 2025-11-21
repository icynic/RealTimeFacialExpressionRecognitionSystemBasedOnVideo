import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix # 导入 confusion_matrix
import numpy as np
from icecream import ic
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import SMOTE




blendshapes_path = "blendshapes/blendshapes.csv"
model_save_path = "models/expression_classifier.tflite"
categories_path = "blendshapes/categories.csv"
random_seed = 42
tf.keras.utils.set_random_seed(random_seed)

# Get categories from the csv file
with open(categories_path, "r") as file:
    content = file.readline().strip()
    categories = [name.capitalize() for name in content.split(",")]

ic(categories)

# Load actual data from the csv file
data = np.loadtxt(blendshapes_path, delimiter=",")
y = data[:, 0]
X = data[:, 1:]


X_train, X_test, y_train, y_test = train_test_split(
    X, y, train_size=0.8, random_state=random_seed, stratify=y
)

ic(X_train.shape, X_test.shape, X.dtype)



smote = SMOTE(sampling_strategy='auto', random_state=random_seed)
X_train, y_train = smote.fit_resample(X_train, y_train)



# Build model
model = tf.keras.models.Sequential(
    [
        tf.keras.layers.Input((52,)),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dense(len(categories)),
    ]
)

model.summary()

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"],
)

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss", patience=20, restore_best_weights=True
)

history = model.fit(
    X_train,
    y_train,
    epochs=600,
    validation_data=(X_test, y_test),
    callbacks=[early_stopping],
)

loss, accuracy = model.evaluate(X_test, y_test)


############################################## plotting loss and accuracy
plt.rcParams['font.family'] = 'SimHei'

# 获取每个 epoch 的准确率和损失
train_accuracy = history.history['accuracy']
val_accuracy = history.history['val_accuracy']
train_loss = history.history['loss'] # 获取训练损失
val_loss = history.history['val_loss'] # 获取验证损失
epochs = range(1, len(train_accuracy) + 1)

# 创建一个图，包含两个子图（一个用于准确率，一个用于损失）
fig1, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

# 绘制准确率曲线
ax1.plot(epochs, train_accuracy, 'bo', label='训练准确度')
ax1.plot(epochs, val_accuracy, 'b', label='评估准确度')
ax1.set_title('训练和评估准确度')
ax1.set_xlabel('周期')
ax1.set_ylabel('准确度')
ax1.legend()
ax1.grid(True) # 添加网格线

# 绘制损失曲线
ax2.plot(epochs, train_loss, 'ro', label='训练损失')
ax2.plot(epochs, val_loss, 'r', label='评估损失')
ax2.set_title('训练和评估损失')
ax2.set_xlabel('周期')
ax2.set_ylabel('损失')
ax2.legend()
ax2.grid(True) # 添加网格线

# 调整子图之间的间距
plt.tight_layout()

# 显示图表
# plt.show() # 先不在这里显示，后面一起显示


############################################## plotting category accuracy

# 在测试集上进行预测
y_pred_logits = model.predict(X_test)
y_pred = np.argmax(y_pred_logits, axis=1) # 获取预测的类别索引

# 统计每个类别的正确预测数量和总数
category_correct = {}
category_total = {}

for i in range(len(categories)):
    category_correct[i] = 0
    category_total[i] = 0

for true_label, pred_label in zip(y_test, y_pred):
    true_label = int(true_label) # 确保标签是整数
    category_total[true_label] += 1
    if true_label == pred_label:
        category_correct[true_label] += 1

# 计算每个类别的准确率
category_accuracy = {}
for i in range(len(categories)):
    if category_total[i] > 0:
        category_accuracy[categories[i]] = category_correct[i] / category_total[i]
    else:
        category_accuracy[categories[i]] = 0 # 如果该类别没有样本，准确率为0

# 将类别名称和准确率提取出来用于绘图
category_names = list(category_accuracy.keys())
accuracy_values = list(category_accuracy.values())

# 创建一个新的图来绘制柱状图
fig2, ax3 = plt.subplots(figsize=(12, 6))

# 绘制柱状图
ax3.bar(category_names, accuracy_values, color='skyblue')
ax3.set_ylabel('准确度')
ax3.set_title('各表情类别识别准确度', y=1.05)
ax3.set_ylim([0, 1]) # 设置Y轴范围为0到1

# 在每个柱状图上显示准确率数值
for i, v in enumerate(accuracy_values):
    ax3.text(i, v + 0.01, f"{v:.2f}", ha='center', va='bottom')

# 旋转X轴标签，避免重叠
plt.xticks(rotation=45, ha='right')

plt.tight_layout() # 调整布局

# # 显示图表
# plt.show() # 先不在这里显示，后面一起显示


############################################## plotting confusion matrix

# 计算混淆矩阵
cm = confusion_matrix(y_test, y_pred)

# 创建一个新的图来绘制混淆矩阵
fig3, ax4 = plt.subplots(figsize=(10, 8))

# 绘制混淆矩阵热力图
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax4,
            xticklabels=categories, yticklabels=categories)

ax4.set_xlabel('预测类别')
ax4.set_ylabel('真实类别')
ax4.set_title('混淆矩阵')

plt.tight_layout() # 调整布局


# 一起显示所有图表
plt.show()


##############################################3

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converted_model = converter.convert()

with open(model_save_path, "wb") as file:
    file.write(converted_model)
