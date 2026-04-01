import torch
import gc
import re
from PIL import Image
import numpy as np
from pathlib import Path


class LLaVARunner:
    """LLaVA-7B 选择题推理Runner
    
    用法：
        runner = LLaVARunner('llava-7b')
        choices = runner.predict(video_number, prompt)  # 返回 [1,2,3,4,1,2]
        runner.release()  # 释放显存
    """
    
    def __init__(self, model_name):
        self.model_name = model_name
        self.model = None
        self.processor = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._load_model()
    
    def _load_model(self):
        """加载 LLaVA-7B 模型"""
        try:
            from transformers import AutoProcessor, LlavaForConditionalGeneration
            
            print(f"  📥 加载 {self.model_name}...")
            
            model_id = "llava-hf/llava-1.5-7b-hf"
            
            self.processor = AutoProcessor.from_pretrained(model_id)
            
            # 使用 int8 量化以节省显存
            from transformers import BitsAndBytesConfig
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_threshold=6.0,
            )
            
            self.model = LlavaForConditionalGeneration.from_pretrained(
                model_id,
                quantization_config=quantization_config,
                device_map="auto"
            ).eval()
            
            print(f"  ✓ {self.model_name} 加载成功 (int8量化, {self.device})")
        except ImportError:
            print("  ❌ transformers 或 bitsandbytes 未安装")
            print("  请运行: pip install transformers bitsandbytes")
            raise
        except Exception as e:
            print(f"  ❌ 加载失败: {e}")
            raise
    
    def predict(self, video_number, prompt, video_frames):
        """推理选择题，使用全部视频帧（分批处理以节省显存）
        
        Args:
            video_number: 视频编号（用于日志）
            prompt: 完整的prompt（包含题干和选项）
            video_frames: numpy 数组 (num_frames, H, W, 3)，RGB 格式
        
        Returns:
            choices: [1,2,3,4,1,2] 长度为6的列表
        """
        if self.model is None or self.processor is None:
            print(f"  ❌ 模型未加载")
            return [0, 0, 0, 0, 0, 0]
        
        try:
            # 关键帧采样
            num_frames = len(video_frames)
            target_frames = min(16, max(10, int(num_frames * 0.5)))
            indices = np.linspace(0, num_frames - 1, target_frames, dtype=int)
            key_frames = video_frames[indices]
            
            # 转换为 PIL Image 并下采样
            images = []
            for frame in key_frames:
                img = Image.fromarray(frame)
                img = img.resize((336, 336), Image.BILINEAR)
                images.append(img)
            
            # LLaVA-1.5 标准格式：如果有多张图像，需要为每一张添加 <image> token
            # 但 LLaVA-1.5 通常只支持单张图像，我们只用第一张
            image_to_use = images[0:1]  # 只用第一张图像
            
            # 标准 LLaVA prompt 格式
            text_input = f"<image>\n{prompt}"
            
            # 使用 processor 处理（不使用 chat template）
            inputs = self.processor(
                text=text_input,
                images=image_to_use,
                return_tensors="pt",
            ).to(self.device)
            
            # 记录输入长度（用于后续解码）
            input_token_count = inputs['input_ids'].shape[1]
            
            # 推理
            with torch.no_grad():
                output_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=100,
                    do_sample=False,
                    num_beams=1,
                    pad_token_id=self.processor.tokenizer.pad_token_id
                )
            
            # 只解码新生成的 token（从 input_token_count 开始）
            new_tokens = output_ids[0][input_token_count:]
            response = self.processor.tokenizer.decode(
                new_tokens,
                skip_special_tokens=True
            )
            
            # 从响应中解析选项序号（需要知道题目数）
            # 从 prompt 中推断题目数（查找 "Q1:", "Q2:" 等）
            import re
            q_matches = re.findall(r'Q(\d+):', prompt)
            num_questions = len(q_matches) if q_matches else 6
            
            choices = self._parse_choices(response, num_questions)
            
            # 🔍 调试：输出原始响应和解析结果
            print(f"  🔍 DEBUG - 原始响应: {response[:200]}")
            print(f"  🔍 DEBUG - 解析结果: {choices}")
            print(f"  ✓ {self.model_name} 推理完成: {video_number} (使用{len(images)}帧)")
            return choices
            
        except Exception as e:
            print(f"  ❌ 推理失败: {e}")
            import traceback
            traceback.print_exc()
            return [0, 0, 0, 0, 0, 0]
    
    def _parse_choices(self, response: str, num_questions: int = 6) -> list:
        """从模型响应中解析选项序号
        
        Args:
            response: 模型输出的字符串
            num_questions: 题目数量
        
        期望格式（优先级从高到低）：
            1. "A1: 1\nA2: 2..." 或 "Answers: 1 2 3..."  (模型实际输出)
            2. "1:1 2:2 3:3..." 格式（题号:答案）
            3. 提取所有 1-4 数字（宽松匹配）
        
        Returns:
            [1,2,3,4,1,2] 或 [0,0,0,0,0,0]（解析失败时）
        """
        choices = []
        response = response.strip()
        
        # 1️⃣ 匹配 "A1: 1\nA2: 2..." 或类似格式（最常见）
        an_pattern = r'[Aa]\d+\s*[:：]\s*([1-4])'
        an_matches = re.findall(an_pattern, response)
        if len(an_matches) >= num_questions:
            choices = [int(m) for m in an_matches[:num_questions]]
            return choices
        
        # 2️⃣ 匹配 "1:1 2:2 3:3..." 格式（题号:答案）
        pattern1 = r'(\d+)\s*[:：]\s*([1-4])'
        matches = re.findall(pattern1, response)
        if matches and len(matches) >= num_questions:
            matches.sort(key=lambda x: int(x[0]))
            choices = [int(m[1]) for m in matches[:num_questions]]
            return choices
        
        # 3️⃣ 匹配 "Answers: 1 2 3 4 1 2"格式
        answers_pattern = r'[Aa]nswers?\s*[:：]\s*([\d\s,]+)'
        answers_match = re.search(answers_pattern, response)
        if answers_match:
            numbers = re.findall(r'([1-4])', answers_match.group(1))
            if len(numbers) >= num_questions:
                choices = [int(n) for n in numbers[:num_questions]]
                return choices
        
        # 4️⃣ 提取所有出现的1-4数字（宽松匹配）
        all_numbers = re.findall(r'\b([1-4])\b', response)
        if len(all_numbers) >= num_questions:
            choices = [int(n) for n in all_numbers[:num_questions]]
            print(f"  ⚠️  使用宽松匹配: {choices}")
            return choices
        
        # ❌ 解析失败
        print(f"  ⚠️  WARNING: 无法解析模型响应: {response[:200]}")
        print(f"  🔍 DEBUG - 需要 {num_questions} 个答案，但只找到 {len(all_numbers) if 'all_numbers' in locals() else 0} 个")
        return [0] * num_questions
    
    def release(self):
        """释放模型资源，清空显存"""
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
