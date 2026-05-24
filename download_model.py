import os
import sys
from transformers import AutoModel, AutoTokenizer

# 模型名称
model_name = "BAAI/bge-base-zh-v1.5"
# 本地存储路径
local_model_path = "./models/bge-base-zh-v1.5"

def download_model():
    print(f"开始下载模型: {model_name}")
    print(f"存储路径: {local_model_path}")
    
    try:
        # 下载模型
        model = AutoModel.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # 保存到本地
        os.makedirs(local_model_path, exist_ok=True)
        model.save_pretrained(local_model_path)
        tokenizer.save_pretrained(local_model_path)
        
        print("✅ 模型下载完成！")
        print(f"模型已保存到: {local_model_path}")
    except Exception as e:
        print(f"❌ 下载失败: {e}")
        sys.exit(1)

if __name__ == "__main__":
    download_model()
