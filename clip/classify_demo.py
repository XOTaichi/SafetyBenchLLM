from transformers import CLIPProcessor, CLIPModel
import torch
from PIL import Image

# 加载 CLIP 模型和处理器
model_name = "openai/clip-vit-base-patch32"
model = CLIPModel.from_pretrained(model_name)
processor = CLIPProcessor.from_pretrained(model_name)

# 加载图像
image = Image.open("pic/test/0_0107.jpg")
# image = Image.open("pic/test/1_2008.jpg")

# 定义分类标签
labels = ["暴力", "打架", "血腥", "封建迷信", "毒品", "低俗", "色情",
          "恐怖", "虐待"]

# 预处理图像和文本
inputs = processor(text=labels, images=image, return_tensors="pt", padding=True)

# 模型推理
with torch.no_grad():
    outputs = model(**inputs)

# 获取图像和文本的相似度分数
logits_per_image = outputs.logits_per_image  # 图像与文本的相似度
probs = logits_per_image.softmax(dim=1)      # 转换为概率

# 解析结果
predicted_label_idx = probs.argmax().item()
predicted_label = labels[predicted_label_idx]
confidence = probs[0][predicted_label_idx].item()

print(f"预测类别: {predicted_label}, 置信度: {confidence:.4f}")