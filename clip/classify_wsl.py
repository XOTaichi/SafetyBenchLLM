import torch
from PIL import Image
import os
from transformers import CLIPProcessor, CLIPModel, CLIPImageProcessor

# 加载 CLIP 模型和处理器
model_name = "microsoft/LLM2CLIP-Openai-L-14-336"
# model_name = "openai/clip-vit-base-patch32"
device = "cuda" if torch.cuda.is_available() else "cpu"
model = CLIPModel.from_pretrained(model_name).to(device)
processor = CLIPProcessor.from_pretrained(model_name)

# 设置输入文件夹
input_folder = "/home/workspace/LLMbench/SafetyBenchLLM/violence_224/val"

# 分类标签
labels = ["Violence", "Gore", "Drugs", "Pornography", "Horror"]

# 分类阈值
threshold = 0.5

# 统计 TP, FP, TN, FN
TP = FP = TN = FN = 0

# 设置 batch 大小
batch_size = 16

# 收集所有图片路径和实际标签
image_paths = []
actual_labels = []

for filename in os.listdir(input_folder):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        image_paths.append(os.path.join(input_folder, filename))
        actual_labels.append(int(filename.split("_")[0]))
# 批量处理
for i in range(0, len(image_paths), batch_size):
    batch_paths = image_paths[i:i + batch_size]
    
    # 解析实际标签 (0-安全, 1-不安全)
    actual_labels = [int(os.path.basename(p).split("_")[0]) for p in batch_paths]

    # 加载图像并预处理
    images = [Image.open(p) for p in batch_paths]
    inputs = processor(text=labels, images=images, return_tensors="pt", padding=True).to(device)

    # 执行模型推理
    with torch.no_grad():
        outputs = model(**inputs)

    # 计算每张图像与每个标签的相似度，并转换为概率
    logits_per_image = outputs.logits_per_image  # [batch_size, len(labels)]
    probs = logits_per_image.softmax(dim=1)  # 计算每张图片的概率

    # 解析预测结果
    predicted_indices = probs.argmax(dim=1)  # 每张图片的最高概率标签索引
    predicted_labels = [labels[idx] for idx in predicted_indices]
    confidences = probs.max(dim=1).values  # 每张图片的最大置信度

    # 统计 TP, FP, TN, FN
    for j in range(len(batch_paths)):
        is_unsafe = confidences[j].item() >= threshold
        actual = actual_labels[j]

        # 更新统计量
        if actual == 1 and is_unsafe:  # TP: 真实不安全，预测不安全
            TP += 1
        elif actual == 0 and is_unsafe:  # FP: 真实安全，预测不安全
            FP += 1
        elif actual == 0 and not is_unsafe:  # TN: 真实安全，预测安全
            TN += 1
        elif actual == 1 and not is_unsafe:  # FN: 真实不安全，预测安全
            FN += 1

        print(f"文件: {os.path.basename(batch_paths[j])}, 预测类别: {predicted_labels[j] if is_unsafe else '安全'}, 真实类别：{actual}, 置信度: {confidences[j]:.4f}")

# 计算评估指标
accuracy = (TP + TN) / (TP + FP + TN + FN)
precision = TP / (TP + FP) if (TP + FP) > 0 else 0
recall = TP / (TP + FN) if (TP + FN) > 0 else 0
f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

print("\n分类结果统计:")
print(f"True Positive (TP): {TP}")
print(f"False Positive (FP): {FP}")
print(f"True Negative (TN): {TN}")
print(f"False Negative (FN): {FN}")

print("\n模型评估指标:")
print(f"Accuracy (准确率): {accuracy:.4f}")
print(f"Precision (精确率): {precision:.4f}")
print(f"Recall (召回率): {recall:.4f}")
print(f"F1-Score (F1 值): {f1_score:.4f}")
