import os

# 使用 HuggingFace 镜像，解决国内连接 huggingface.co 超时问题
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

from transformers import HfArgumentParser

from ltx_video.inference import infer, InferenceConfig


def main():
    parser = HfArgumentParser(InferenceConfig)
    config = parser.parse_args_into_dataclasses()[0]
    infer(config=config)


if __name__ == "__main__":
    main()
