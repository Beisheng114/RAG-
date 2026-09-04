"""
模型下载脚本

用法：
    python download_model.py            # 下载嵌入模型 BAAI/bge-base-zh-v1.5（必需）
    python download_model.py --rerank   # 下载精排模型 BAAI/bge-reranker-base（可选，用于 Rerank 精排）
    python download_model.py --all      # 下载全部模型
"""
import argparse
import os
import sys

from transformers import AutoModel, AutoTokenizer, AutoModelForSequenceClassification

# 模型清单：名称 -> (HuggingFace ID, 本地路径, 是否需要 SequenceClassification 头)
MODELS = {
    "embedding": ("BAAI/bge-base-zh-v1.5", "./models/bge-base-zh-v1.5", False),
    "rerank": ("BAAI/bge-reranker-base", "./models/bge-reranker-base", True),
}


def download_model(key: str) -> None:
    hf_id, local_path, need_cls_head = MODELS[key]
    print(f"开始下载模型: {hf_id}")
    print(f"存储路径: {local_path}")

    try:
        if need_cls_head:
            # CrossEncoder/Reranker 需要 SequenceClassification 头
            model = AutoModelForSequenceClassification.from_pretrained(hf_id)
        else:
            model = AutoModel.from_pretrained(hf_id)
        tokenizer = AutoTokenizer.from_pretrained(hf_id)

        os.makedirs(local_path, exist_ok=True)
        model.save_pretrained(local_path)
        tokenizer.save_pretrained(local_path)

        print(f"✅ 模型下载完成，已保存到 {local_path}")
    except Exception as e:
        print(f"❌ 下载失败: {e}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="模型下载脚本")
    parser.add_argument("--rerank", action="store_true", help="下载精排模型 BAAI/bge-reranker-base")
    parser.add_argument("--all", action="store_true", help="下载全部模型（嵌入+精排）")
    args = parser.parse_args()

    if args.all:
        download_model("embedding")
        download_model("rerank")
    elif args.rerank:
        download_model("rerank")
    else:
        # 默认：下载嵌入模型
        download_model("embedding")


if __name__ == "__main__":
    main()
