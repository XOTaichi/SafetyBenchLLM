import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel, AutoTokenizer, AutoModelForSequenceClassification

# 1️⃣ 加载 CLIP 模型
device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model_name = "openai/clip-vit-base-patch32"#"openai/clip-vit-large-patch14"
clip_model = CLIPModel.from_pretrained(clip_model_name).to(device)
clip_processor = CLIPProcessor.from_pretrained(clip_model_name)

# 2️⃣ 加载 LLM（用于分类）
llm_name = "/data/models/Qwen/Qwen2.5-1.5B-Instruct/"
tokenizer = AutoTokenizer.from_pretrained(llm_name)
llm_model = AutoModelForSequenceClassification.from_pretrained(llm_name).to(device)

# 3️⃣ 提取图像和文本特征
def extract_clip_features(image_path, text):
    image = Image.open(image_path).convert("RGB")
    inputs = clip_processor(text=[text], images=image, return_tensors="pt").to(device)
    outputs = clip_model(**inputs)

    # 图像特征 (image_embedding) 和 文本特征 (text_embedding)
    image_emb = outputs.image_embeds
    text_emb = outputs.text_embeds

    # 图文特征融合（例如拼接、求和、注意力等方法）
    fused_emb = torch.cat((image_emb, text_emb), dim=1)  # 拼接方式
    return fused_emb

# 4️⃣ 进行不良内容分类
def classify_image_with_llm(image_path, prompt):
    # 提取融合后的特征
    fused_embedding = extract_clip_features(image_path, prompt)

    # 将特征作为输入，适配 LLM 分类
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    outputs = llm_model(**inputs)

    # 获取分类结果
    logits = outputs.logits
    probs = torch.softmax(logits, dim=-1)
    print("分类概率: ", probs)
    return torch.argmax(probs, dim=-1).item()

# 5️⃣ 进行测试
image_path = "test.jpg"
prompt = "A potentially unsafe image related to violence, pornography, or abuse."
result = classify_image_with_llm(image_path, prompt)

if result == 1:
    print("⚠️ 该图片可能包含不良内容！")
else:
    print("✅ 该图片为正常内容。")
