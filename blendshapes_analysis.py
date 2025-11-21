import matplotlib.pyplot as plt
import numpy as np
import re

plt.rcParams['font.family'] = 'SimHei'


# 提取数据
blendshape_times = []
classifying_times = []

# 使用上传的文件内容
with open('paste.txt', 'r') as file:
    lines = file.readlines()
    
for line in lines:
    blendshape_match = re.search(r'Blendshapes extracting time\(ms\):\s+(\d+\.\d+)', line)
    if blendshape_match:
        blendshape_times.append(float(blendshape_match.group(1)))
    
    classifying_match = re.search(r'classifying time:\s+(\d+\.\d+)', line)
    if classifying_match:
        classifying_times.append(float(classifying_match.group(1)))

# 创建图表
plt.figure(figsize=(15, 10))

# 1. Blendshapes提取时间图表
plt.subplot(2, 2, 1)
plt.plot(blendshape_times, 'b-', linewidth=1)
plt.xlabel('样本数', fontsize=12)
plt.ylabel('时间 (毫秒)', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.title('Blendshapes提取时间 (毫秒)', fontsize=14)
plt.text(0.5, -0.35, '图3.12.1',
         horizontalalignment='center',
         fontsize=14,
         transform=plt.gca().transAxes)

# 标注第一个较高的值
# if len(blendshape_times) > 0 and max(blendshape_times) > 40:
#     plt.annotate(f'{blendshape_times[0]:.2f} ms', 
#                  xy=(0, blendshape_times[0]),
#                  xytext=(10, blendshape_times[0] + 2),
#                  arrowprops=dict(facecolor='black', shrink=0.05, width=1.5))

# 2. 分类时间图表
plt.subplot(2, 2, 2)
plt.plot(classifying_times, 'r-', linewidth=1)
plt.xlabel('样本数', fontsize=12)
plt.ylabel('时间 (毫秒)', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.title('分类时间 (毫秒)', fontsize=14)
plt.text(0.5, -0.35, '图3.12.2',
         horizontalalignment='center',
         fontsize=14,
         transform=plt.gca().transAxes)

# 标注第一个较高的值
# if len(classifying_times) > 0 and max(classifying_times) > 7:
#     plt.annotate(f'{classifying_times[0]:.2f} s', 
#                  xy=(0, classifying_times[0]),
#                  xytext=(10, classifying_times[0] - 1),
#                  arrowprops=dict(facecolor='black', shrink=0.05, width=1.5))

# 3. Blendshapes提取时间直方图
plt.subplot(2, 2, 3)
blendshape_times = [t for t in blendshape_times if t < 40]  # 过滤掉异常高值
plt.hist(blendshape_times, bins=20, color='skyblue', edgecolor='black', alpha=0.7)
plt.xlabel('时间 (毫秒)', fontsize=12)
plt.ylabel('频率', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.title('Blendshapes提取时间分布 (排除异常值)', fontsize=14)
plt.text(0.5, -0.35, '图3.12.3',
         horizontalalignment='center',
         fontsize=14,
         transform=plt.gca().transAxes)

# 添加平均值线
mean_blendshape = np.mean(blendshape_times)
plt.axvline(mean_blendshape, color='b', linestyle='dashed', linewidth=1)
plt.text(mean_blendshape + 0.5, plt.ylim()[1] * 0.9, f'平均值: {mean_blendshape:.2f} ms', rotation=0)

# 4. 分类时间直方图（排除异常值以便更好地观察主要分布）
plt.subplot(2, 2, 4)
filtered_classifying = [t for t in classifying_times if t < 5]  # 过滤掉异常高值
plt.hist(filtered_classifying, bins=20, color='salmon', edgecolor='black', alpha=0.7)
plt.xlabel('时间 (毫秒)', fontsize=12)
plt.ylabel('频率', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.title('分类时间分布 (排除异常值)', fontsize=14)
plt.text(0.5, -0.35, '图3.12.4',
         horizontalalignment='center',
         fontsize=14,
         transform=plt.gca().transAxes)

# 添加平均值线（排除异常值后）
if filtered_classifying:
    mean_classifying = np.mean(filtered_classifying)
    plt.axvline(mean_classifying, color='r', linestyle='dashed', linewidth=1)
    plt.text(mean_classifying + 0.05, plt.ylim()[1] * 0.9, f'平均值: {mean_classifying:.2f} ms', rotation=0)

# 5. 添加统计数据表
# plt.figtext(0.5, 0.01, f"""
# 统计数据:
# Blendshapes提取时间: 平均值 = {np.mean(blendshape_times):.2f} ms, 最小值 = {min(blendshape_times):.2f} ms, 最大值 = {max(blendshape_times):.2f} ms
# 分类时间: 平均值 = {np.mean(classifying_times):.2f} ms, 最小值 = {min(classifying_times):.2f} ms, 最大值 = {max(classifying_times):.2f} ms
# """,
#            ha='center', fontsize=10, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.2))

plt.tight_layout()
plt.subplots_adjust(top=0.92, bottom=0.2, left=0.1, hspace=0.7)
# plt.suptitle('Blendshapes提取和分类时间分析', fontsize=16)

plt.show()
