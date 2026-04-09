import gc
import re
from typing import List

import numpy as np
import torch
from PIL import Image


class LLaVAOneVisionRunner:
    """LLaVA-OneVision 推理 Runner

    - 尝试直接视频处理（processor.videos），若不支持则退化为帧序列处理。
    - 默认最多使用50帧，均匀采样。
    """

    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model = None
        self.processor = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._load_model()

    def _load_model(self):
        try:
            import os
            from transformers import AutoProcessor, LlavaOnevisionForConditionalGeneration

            model_id = "llava-hf/llava-onevision-qwen2-7b-ov-hf"
            cache_dir = os.getenv('HF_LOCAL_DIR_ROOT', None)
            print(f"  📥 加载 {self.model_name} ({model_id}) ...")

            self.processor = AutoProcessor.from_pretrained(model_id, cache_dir=cache_dir)
            self.model = LlavaOnevisionForConditionalGeneration.from_pretrained(
                model_id,
                torch_dtype=torch.float16,
                device_map="auto",
                cache_dir=cache_dir,
            ).eval()

            print(f"  ✓ {self.model_name} 加载成功")
        except Exception as e:
            print(f"  ❌ 加载失败: {e}")
            raise

    def predict(self, video_number: str, prompt: str, video_frames: np.ndarray, num_options: int = 4, expected_count: int = 1) -> List[int]:
        if self.model is None or self.processor is None:
            print("  ❌ 模型未加载")
            return [0] * expected_count

        try:
            # 均匀采样至最多50帧
            total_frames = len(video_frames)
            target_frames = min(32, total_frames) if total_frames > 0 else 1
            indices = np.linspace(0, total_frames - 1, target_frames, dtype=int) if total_frames > 0 else [0]
            sampled_frames = video_frames[indices]
            pil_frames = [Image.fromarray(f).convert("RGB") for f in sampled_frames]

            # 首选：使用官方 chat template + 视频输入；系统消息强约束只输出数字
            inputs = None
            try:
                messages = [
                    {
                        "role": "system",
                        "content": "You are an expert traffic safety analyst. Answer the question by outputting ONLY ONE single option number. No explanations, no extra text.",
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "video"},
                            {"type": "text", "text": prompt},
                        ],
                    },
                ]
                chat_text = self.processor.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    tokenize=False,
                )
                inputs = self.processor(
                    text=[chat_text],
                    videos=[pil_frames],
                    return_tensors="pt",
                )
            except Exception:
                inputs = None

            # 退化：多帧序列输入（<image> 占位符），仍然用 chat template 文本
            if inputs is None:
                image_placeholders = "\n".join(["<image>" for _ in pil_frames])
                text_for_images = f"{image_placeholders}\n{prompt}"
                messages = [
                    {
                        "role": "system",
                        "content": "You are an expert traffic safety analyst. Answer the question by outputting ONLY ONE single option number. No explanations, no extra text.",
                    },
                    {
                        "role": "user",
                        "content": text_for_images,
                    },
                ]
                chat_text = self.processor.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    tokenize=False,
                )
                inputs = self.processor(
                    text=[chat_text],
                    images=pil_frames,
                    return_tensors="pt",
                )

            inputs = inputs.to(self.device)
            input_token_count = inputs["input_ids"].shape[1]

            # 推理
            with torch.no_grad():
                output_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=128,
                    do_sample=False,
                    num_beams=1,
                    pad_token_id=self.processor.tokenizer.pad_token_id,
                )

            new_tokens = output_ids[0][input_token_count:]
            response = self.processor.tokenizer.decode(new_tokens, skip_special_tokens=True)

            choices = self._parse_single_choice(response, num_options)

            print(f"  🔍 DEBUG - 原始响应: {response[:200]}")
            print(f"  🔍 DEBUG - 解析结果: {choices}")
            print(f"  ✓ {self.model_name} 推理完成: {video_number} (使用{target_frames}帧)")
            return choices
        except Exception as e:
            print(f"  ❌ 推理失败: {e}")
            return [0] * expected_count

    def _parse_single_choice(self, response: str, num_options: int) -> List[int]:
        max_opt = str(num_options)
        opt_char_class = f'[1-{max_opt}]'
        
        # 1. 类似 X: [answer] 或 Answer: [answer]
        pattern1 = rf'(?:答案|选项|option|answer)\s*[：:]*\s*({opt_char_class})'
        matches1 = re.findall(pattern1, response.lower())
        if matches1:
            return [int(matches1[0])]
            
        # 2. 只有数字
        numbers = re.findall(rf'\b({opt_char_class})\b', response)
        if numbers:
            return [int(numbers[0])]
            
        return [0]

    def release(self):
        if self.model is not None:
            del self.model
            self.model = None
        if self.processor is not None:
            del self.processor
            self.processor = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"  ♻️ {self.model_name} 资源已释放")
        print(f"  ♻️ {self.model_name} 资源已释放")
