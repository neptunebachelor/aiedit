from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import dashscope
import requests
from dashscope import MultiModalConversation
from dotenv import load_dotenv
from requests.exceptions import SSLError


DEFAULT_MODEL = "qwen3.6-plus"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Minimal MVP for sending a local video directly to Qwen via DashScope."
    )
    parser.add_argument("--video", required=True, help="Path to the local video file.")
    parser.add_argument("--prompt", help="Prompt to send with the video.")
    parser.add_argument("--prompt-file", help="Read the prompt from a UTF-8 text file.")
    parser.add_argument(
        "--prompt-stdin",
        action="store_true",
        help="Read the prompt from standard input.",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"DashScope model name. Default: {DEFAULT_MODEL}")
    parser.add_argument("--output", help="Optional path to save the raw response JSON.")
    parser.add_argument("--ca-bundle", help="Path to a custom CA bundle PEM file for HTTPS verification.")
    parser.add_argument("--no-proxy", action="store_true", help="Ignore HTTP(S) proxy environment variables.")
    parser.add_argument(
        "--insecure",
        action="store_true",
        help="Disable HTTPS certificate verification. Use only for local debugging.",
    )
    args = parser.parse_args()

    prompt_sources = [bool(args.prompt), bool(args.prompt_file), bool(args.prompt_stdin)]
    if sum(prompt_sources) != 1:
        parser.error("Provide exactly one of --prompt, --prompt-file, or --prompt-stdin.")
    return args


def to_file_url(path: Path) -> str:
    resolved = path.expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"Video file does not exist: {resolved}")
    if not resolved.is_file():
        raise ValueError(f"Video path is not a file: {resolved}")
    return f"file://{resolved}"


def extract_text(response: Any) -> str:
    try:
        content = response.output.choices[0].message.content
    except (AttributeError, IndexError, KeyError, TypeError):
        return json.dumps(response, ensure_ascii=False, indent=2, default=str)

    if isinstance(content, list):
        text_parts: list[str] = []
        for item in content:
            if isinstance(item, dict) and "text" in item:
                text_parts.append(str(item["text"]))
        if text_parts:
            return "\n".join(text_parts)
    return json.dumps(content, ensure_ascii=False, indent=2, default=str)


def read_prompt(args: argparse.Namespace) -> str:
    if args.prompt is not None:
        return args.prompt
    if args.prompt_file is not None:
        prompt_path = Path(args.prompt_file).expanduser().resolve()
        if not prompt_path.exists():
            raise FileNotFoundError(f"Prompt file does not exist: {prompt_path}")
        if not prompt_path.is_file():
            raise ValueError(f"Prompt path is not a file: {prompt_path}")
        return prompt_path.read_text(encoding="utf-8")
    return sys.stdin.read()


def configure_network(args: argparse.Namespace) -> None:
    if args.ca_bundle:
        ca_bundle = str(Path(args.ca_bundle).expanduser().resolve())
        os.environ["REQUESTS_CA_BUNDLE"] = ca_bundle
        os.environ["CURL_CA_BUNDLE"] = ca_bundle

    if args.no_proxy:
        for name in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
            os.environ.pop(name, None)
        os.environ["NO_PROXY"] = "*"
        os.environ["no_proxy"] = "*"

    if args.insecure:
        original_request = requests.sessions.Session.request

        def insecure_request(self: requests.Session, method: str, url: str, **kwargs: Any) -> requests.Response:
            kwargs.setdefault("verify", False)
            return original_request(self, method, url, **kwargs)

        requests.sessions.Session.request = insecure_request


def print_status(message: str) -> None:
    print(message, flush=True)


def build_ssl_help() -> str:
    return (
        "SSL 握手失败，通常不是脚本逻辑错误，而是上传到 DashScope OSS 时被代理、证书链或网络策略拦截。\n"
        "可以依次尝试：\n"
        "1. 加 `--no-proxy`，绕过系统里的 HTTP(S) 代理。\n"
        "2. 如果网络有 HTTPS 检查，导出代理根证书 PEM 后加 `--ca-bundle <pem路径>`。\n"
        "3. 仅本地排查时可临时加 `--insecure` 跳过证书校验。\n"
        "4. 换一个不走代理的网络再试。\n"
        "目标上传域名是：dashscope-file-mgr.oss-cn-beijing.aliyuncs.com"
    )


def main() -> int:
    load_dotenv()
    args = parse_args()
    configure_network(args)

    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        print("Missing environment variable: DASHSCOPE_API_KEY", file=sys.stderr)
        return 1

    print_status("正在检查视频文件...")
    video_path = Path(args.video).expanduser().resolve()
    video_url = to_file_url(video_path)

    print_status("正在读取提示词...")
    prompt = read_prompt(args)

    try:
        print_status(f"准备上传视频到 Qwen: {video_path}")
        print_status("正在上传视频并等待模型返回，这一步可能需要几十秒到几分钟...")
        response = MultiModalConversation.call(
            api_key=api_key,
            model=args.model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"video": video_url},
                        {"text": prompt},
                    ],
                }
            ],
        )
    except SSLError as exc:
        print(build_ssl_help(), file=sys.stderr)
        print(f"\n原始错误: {exc}", file=sys.stderr)
        return 2

    if args.output:
        output_path = Path(args.output).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        print_status(f"正在保存原始响应到: {output_path}")
        output_path.write_text(
            json.dumps(response, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )

    print_status("模型已返回结果：")
    print(extract_text(response))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
