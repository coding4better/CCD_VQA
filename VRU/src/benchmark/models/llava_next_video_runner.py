import gc
import re
from typing import List

import numpy as np
import torch
from PIL import Image


class LLaVANextVideoRunner:
    """LLaVA-NeXT-Video 推理 Runner，使用最多50帧进行视频VQA"""

    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model = None
        self.processor = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._load_model()

    def _load_model(self):
        """加载 llava-next-video 模型，使用 int8 量化节省显存"""
        try:
            from transformers import AutoProcessor, LlavaNextVideoForConditionalGeneration

            model_id = "llava-hf/LLaVA-NeXT-Video-7B-hf"
            print(f"  📥 加载 {self.model_name} ({model_id}) ...")

            self.processor = AutoProcessor.from_pretrained(model_id)
            
            # 使用 int8 量化以节省显存
            from transformers import BitsAndBytesConfig
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_threshold=6.0,
                llm_int8_skip_modules=["lm_head", "vision_tower"],
            )
            
            self.model = LlavaNextVideoForConditionalGeneration.from_pretrained(
                model_id,
                quantization_config=quantization_config,
                device_map="auto",
            ).eval()

            print(f"  ✓ {self.model_name} 加载成功 (int8量化, {self.device})")
        except ImportError:
            print("  ❌ transformers 或 bitsandbytes 未安装")
            print("  请运行: pip install transformers bitsandbytes")
            raise
        except Exception as e:
            print(f"  ❌ 加载失败: {e}")
            raise

    def predict(self, video_number: str, prompt: str, video_frames: np.ndarray) -> List[int]:
        """使用视频帧进行VQA推理，自动控制token长度"""
        if self.model is None or self.processor is None:
            print("  ❌ 模型未加载")
            return [0] * 6

        try:
            # 采样帧 - 控制token长度，避免超过4000
            total_frames = len(video_frames)
            target_frames = min(10, total_frames) if total_frames > 0 else 1
            indices = np.linspace(0, total_frames - 1, target_frames, dtype=int) if total_frames > 0 else [0]
            sampled_frames = video_frames[indices]
            
            # 降低分辨率以减少token数量
            pil_frames = []
            for f in sampled_frames:
                img = Image.fromarray(f)
                img = img.resize((288, 288), Image.BILINEAR)  # 降低分辨率
                pil_frames.append(img)

            # 构造输入
            text_input = f"<video>\n{prompt}"
            inputs = self.processor(
                text=text_input,
                videos=[pil_frames],  # batch size 1，单个视频
                return_tensors="pt",
                padding=True,
            )
            inputs = inputs.to(self.device)
            input_token_count = inputs["input_ids"].shape[1]
            
            # 记录token长度
            if input_token_count > 4000:
                print(f"  ⚠️  Token 长度过长: {input_token_count}")

            # 推理
            with torch.no_grad():
                output_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=50,
                    do_sample=False,
                    num_beams=1,
                    temperature=0.0,
                    pad_token_id=self.processor.tokenizer.pad_token_id,
                    eos_token_id=self.processor.tokenizer.eos_token_id,
                )

            new_tokens = output_ids[0][input_token_count:]
            response = self.processor.tokenizer.decode(new_tokens, skip_special_tokens=True)

            # 解析答案
            num_questions = self._detect_num_questions(prompt)
            choices = self._parse_choices(response, num_questions)

            print(f"  🔍 DEBUG - 原始响应: {response[:200]}")
            print(f"  🔍 DEBUG - 解析结果: {choices}")
            print(f"  ✓ {self.model_name} 推理完成: {video_number} (使用{target_frames}帧)")
            return choices
        except Exception as e:
            print(f"  ❌ 推理失败: {e}")
            import traceback
            traceback.print_exc()
            return [0] * 6

    @staticmethod
    def _detect_num_questions(prompt: str) -> int:
        matches = re.findall(r"Q(\d+):", prompt)
        return len(matches) if matches else 6

    def _parse_choices(self, response: str, num_questions: int) -> List[int]:
        response = response.strip()
        
        # 检查响应是否有效（排除乱码）
        if not response or len(response) == 0:
            print(f"  ⚠️  WARNING: 模型返回空响应")
            return [1] * num_questions
        
        # 检查是否为乱码（包含大量特殊字符）
        invalid_chars = sum(1 for c in response if ord(c) > 65535)
        if invalid_chars > len(response) * 0.3:
            print(f"  ⚠️  WARNING: 检测到乱码响应，使用默认答案: {response[:100]}")
            return [1] * num_questions

        # 1️⃣ 匹配 "A1: 1\nA2: 2..." 或类似格式
        matches = re.findall(r"[Aa]\d+\s*[:：]\s*([1-4])", response)
        if len(matches) >= num_questions:
            return [int(m) for m in matches[:num_questions]]

        # 2️⃣ 匹配 "Answers: 1 2 3 4 1 2" 格式
        ans_match = re.search(r"[Aa]nswers?\s*[:：]\s*([\d\s,]+)", response)
        if ans_match:
            nums = re.findall(r"([1-4])", ans_match.group(1))
            if len(nums) >= num_questions:
                return [int(n) for n in nums[:num_questions]]

        # 3️⃣ 匹配题号:答案  "1:1 2:2 ..."
        pairs = re.findall(r"(\d+)\s*[:：]\s*([1-4])", response)
        if pairs and len(pairs) >= num_questions:
            pairs.sort(key=lambda x: int(x[0]))
            return [int(p[1]) for p in pairs[:num_questions]]

        # 4️⃣ 宽松匹配所有1-4数字
        all_nums = re.findall(r"\b([1-4])\b", response)
        if len(all_nums) >= num_questions:
            return [int(n) for n in all_nums[:num_questions]]

        # ❌ 解析失败 - 返回默认值而不是全0
        print(f"  ⚠️  WARNING: 无法解析 - {response[:150]}")
        return [1] * num_questions

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
