"""
模型下载脚本

用法：
    python download_model.py                     # 下载嵌入模型（bge-base-zh-v1.5，必需）
    python download_model.py --rerank            # 下载精排模型（bge-reranker-base，可选）
    python download_model.py --all               # 下载全部模型
    python download_model.py --source modelscope # 显式指定下载源

下载源（--source）：
    auto（默认）   先探测 HuggingFace（含 HF_ENDPOINT 环境变量覆盖，如
                   https://hf-mirror.com）可达性，不可达直接走魔搭，
                   避免触发 transformers 内置的长时间重试；HF 下载失败也会回退魔搭
    huggingface   仅 HuggingFace（尊重 HF_ENDPOINT）
    modelscope    仅阿里魔搭（国内/企业网络屏蔽 huggingface.co 时的替代源，
                   两模型均有官方镜像，API 直连无需额外依赖）
"""
import argparse
import os
import sys

import requests

MODELSCOPE_BASE = "https://www.modelscope.cn"

# 模型清单：模型ID（HF/魔搭通用） -> 本地存储路径
MODELS = {
    "embedding": ("BAAI/bge-base-zh-v1.5", "./models/bge-base-zh-v1.5"),
    "rerank": ("BAAI/bge-reranker-base", "./models/bge-reranker-base"),
}


def _hf_endpoint() -> str:
    return os.getenv("HF_ENDPOINT") or "https://huggingface.co"


def _hf_reachable(timeout: int = 5) -> bool:
    """探测 HuggingFace 端点可达性（失败通常 <1s，避免长重试）"""
    try:
        requests.head(_hf_endpoint(), timeout=timeout, allow_redirects=True)
        return True
    except Exception:
        return False


def _download_via_hf(model_id: str, local_path: str) -> None:
    from transformers import (
        AutoModel, AutoTokenizer, AutoModelForSequenceClassification,
    )
    print(f"从 HuggingFace({_hf_endpoint()}) 下载: {model_id}")
    # CrossEncoder/Reranker 需要 SequenceClassification 头
    if "reranker" in model_id:
        model = AutoModelForSequenceClassification.from_pretrained(model_id)
    else:
        model = AutoModel.from_pretrained(model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    os.makedirs(local_path, exist_ok=True)
    model.save_pretrained(local_path)
    tokenizer.save_pretrained(local_path)
    print(f"✅ 已保存到 {local_path}")


def _modelscope_list_files(model_id: str):
    """获取魔搭模型仓库的文件清单"""
    resp = requests.get(
        f"{MODELSCOPE_BASE}/api/v1/models/{model_id}/repo/files",
        params={"Recursive": "true"},
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()
    files = (data.get("Data") or {}).get("Files") or []
    blobs = [f for f in files if f.get("Type") == "blob"]
    if not blobs:
        raise RuntimeError(f"ModelScope 返回空文件列表: {model_id}")
    return blobs


def _download_via_modelscope(model_id: str, local_path: str) -> None:
    """从魔搭逐文件下载模型（纯 requests，保留仓库相对路径结构）"""
    print(f"从 ModelScope({MODELSCOPE_BASE}) 下载: {model_id}")
    blobs = _modelscope_list_files(model_id)
    total = sum(f.get("Size", 0) for f in blobs)
    print(f"  共 {len(blobs)} 个文件，约 {total / 1024 / 1024:.1f} MB")

    os.makedirs(local_path, exist_ok=True)
    for f in blobs:
        rel = f["Path"]
        dest = os.path.join(local_path, rel)
        os.makedirs(os.path.dirname(dest) or local_path, exist_ok=True)
        url = f"{MODELSCOPE_BASE}/models/{model_id}/resolve/master/{rel}"
        size = f.get("Size", 0)
        print(f"  下载 {rel} ({size / 1024 / 1024:.1f} MB) ...", flush=True)
        with requests.get(url, stream=True, timeout=300) as r:
            r.raise_for_status()
            done = 0
            with open(dest, "wb") as out:
                for chunk in r.iter_content(chunk_size=1024 * 1024):
                    out.write(chunk)
                    done += len(chunk)
        if size and done != size:
            print(f"  ⚠️ {rel} 大小不符（预期 {size}，实际 {done}），请重试")
    print(f"✅ 已保存到 {local_path}")


def download_model(key: str, source: str = "auto") -> None:
    model_id, local_path = MODELS[key]
    print(f"开始下载模型: {model_id}")
    print(f"存储路径: {local_path}")

    use_modelscope = source == "modelscope" or (source == "auto" and not _hf_reachable())
    if source == "auto":
        print(f"下载源探测: HuggingFace {'不可达' if use_modelscope else '可达'}")

    try:
        if use_modelscope:
            _download_via_modelscope(model_id, local_path)
        else:
            _download_via_hf(model_id, local_path)
    except Exception as e:
        if source == "auto" and not use_modelscope:
            print(f"⚠️ HuggingFace 下载失败: {e}")
            print("   回退到 ModelScope ...")
            try:
                _download_via_modelscope(model_id, local_path)
            except Exception as e2:
                print(f"❌ ModelScope 也下载失败: {e2}")
                sys.exit(1)
        else:
            print(f"❌ 下载失败: {e}")
            sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="模型下载脚本")
    parser.add_argument("--rerank", action="store_true", help="下载精排模型 BAAI/bge-reranker-base")
    parser.add_argument("--all", action="store_true", help="下载全部模型（嵌入+精排）")
    parser.add_argument(
        "--source", choices=["auto", "huggingface", "modelscope"], default="auto",
        help="下载源（默认 auto：探测 HF，不可达走魔搭）",
    )
    args = parser.parse_args()

    if args.all:
        download_model("embedding", args.source)
        download_model("rerank", args.source)
    elif args.rerank:
        download_model("rerank", args.source)
    else:
        # 默认：下载嵌入模型
        download_model("embedding", args.source)


if __name__ == "__main__":
    main()
