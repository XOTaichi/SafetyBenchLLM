from evaluation.target.qwen import Qwen
import numpy as np
qwen = Qwen(model_path="/root/.cache/huggingface/hub/models--Qwen--Qwen2.5-Coder-7B-Instruct/snapshots/0eb6b1ed2d0c4306bc637d09ecef51e59d3dfe05/")
response, decoded_probs = qwen.get_logit("What is the capital of France?", top_k=10)

# 打印模型的生成响应
print("Response:", response)

# # 打印每个位置的解码概率和 token
for i, step in enumerate(decoded_probs):
    print(f"Step {i + 1}:")
    for prob, token in zip(step["top_k_probs"], step["top_k_tokens"]):
        print(f"  Token: {token}, Probability: {prob:.4f}")