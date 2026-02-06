import os

# 使用 HuggingFace 镜像，解决国内连接 huggingface.co 超时问题
# IMPORTANT: must be set before importing transformers/huggingface_hub.
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_ENDPOINT", "https://hf-mirror.com")

from transformers import HfArgumentParser  # noqa: E402

from ltx_video.inference import infer, InferenceConfig  # noqa: E402


def main():
    parser = HfArgumentParser(InferenceConfig)
    config = parser.parse_args_into_dataclasses()[0]
    infer(config=config)


if __name__ == "__main__":
    main()
