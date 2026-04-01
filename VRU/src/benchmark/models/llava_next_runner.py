import torch
import gc
import re
from PIL import Image
import numpy as np
from pathlib import Path


class LLaVANextRunner:
    """LLaVA-NeXT-7B 选择题推理Runner
    
    改进版本的LLaVA，具有更好的多模态对齐和视觉理解能力
    
    用法：
        runner = LLaVANextRunner('llava-next-7b')
        choices = runner.predict(video_number, prompt, video_frames)
        runner.release()
    """
    
    def __init__(self, model_name):
        self.model_name = model_name
        self.model = None
        self.processor = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._load_model()
    
    def _load_model(self):
        """加载 LLaVA-NeXT-7B 模型"""
        try:
            from transformers import AutoProcessor, LlavaNextForConditionalGeneration
            
            print(f"  📥 加载 {self.model_name}...")
            
            # LLaVA-NeXT官方模型
            model_id = "llava-hf/llava-v1.6-vicuna-7b-hf"
            
            self.processor = AutoProcessor.from_pretrained(model_id)
            
            # 使用 int8 量化以节省显存
            from transformers import BitsAndBytesConfig
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_threshold=6.0,
                llm_int8_skip_modules=["lm_head", "vision_tower"],  # 跳过某些层的量化
            )
            
            self.model = LlavaNextForConditionalGeneration.from_pretrained(
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
        """推理选择题，使用关键帧进行多模态理解
        
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
            # 关键帧采样 - 减少帧数以避免超过token限制
            num_frames = len(video_frames)
            target_frames = 4  # 减少帧数以节省 token
            
            indices = np.linspace(0, num_frames - 1, target_frames, dtype=int)
            key_frames = video_frames[indices]
            
            # 转换为 PIL Image
            images = []
            for frame in key_frames:
                img = Image.fromarray(frame)
                # 降低分辨率以减少token数量
                img = img.resize((224, 224), Image.BILINEAR)
                images.append(img)
            
            # 使用采样的帧进行多模态理解
            image_to_use = images
            
            # 标准格式 - 多图像
            text_input = f"<image>\n" * len(image_to_use) + prompt
            
            # 处理输入
            inputs = self.processor(
                text=text_input,
                images=image_to_use,
                return_tensors="pt",
            ).to(self.device)
            
            # 记录输入长度
            input_token_count = inputs['input_ids'].shape[1]
            token_length = input_token_count
            
            if token_length > 4000:
                print(f"  ⚠️  Token 长度过长: {token_length}")
            
            # 推理
            with torch.no_grad():
                output_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=50,
                    do_sample=False,
                    num_beams=1,
                    pad_token_id=self.processor.tokenizer.pad_token_id
                )
            
            # 只解码新生成的 token
            new_tokens = output_ids[0][input_token_count:]
            response = self.processor.tokenizer.decode(
                new_tokens,
                skip_special_tokens=True
            )
            
            # 从响应中解析选项序号
            choices = self._parse_choices(response)
            
            # 调试输出
            print(f"  🔍 DEBUG - 原始响应: {response[:200]}")
            print(f"  🔍 DEBUG - 解析结果: {choices}")
            print(f"  ✓ {self.model_name} 推理完成: {video_number} (使用{len(images)}帧)")
            return choices
            
        except Exception as e:
            print(f"  ❌ 推理失败: {e}")
            import traceback
            traceback.print_exc()
            return [1, 1, 1, 1, 1, 1]
    
    def _parse_choices(self, response: str) -> list:
        """从模型响应中解析选项序号
        
        期望格式（优先级从高到低）：
            1. "A1: 1\nA2: 2..." 或 "Answers: 1 2 3..."  (模型实际输出)
            2. "Answer: X" 格式
            3. "1:1 2:2 3:3..." 格式（题号:答案）
            4. "1, 2, 3, 4..." 逗号分隔
        
        Returns:
            [1,2,3,4,1,2] 或 [0,0,0,0,0,0]（解析失败时）
        """
        choices = []
        response = response.strip()
        
        # 1️⃣ 匹配 "A1: 1\nA2: 2..." 或类似格式
        an_pattern = r'[Aa]\d+\s*[:：]\s*([1-4])'
        an_matches = re.findall(an_pattern, response)
        if len(an_matches) >= 6:
            choices = [int(m) for m in an_matches[:6]]
            return choices
        
        # 2️⃣ 匹配 "Answers: 1 2 3 4 1 2"格式
        answers_pattern = r'[Aa]nswers?\s*[:：]\s*([\d\s,]+)'
        answers_match = re.search(answers_pattern, response)
        if answers_match:
            numbers = re.findall(r'([1-4])', answers_match.group(1))
            if len(numbers) >= 6:
                choices = [int(n) for n in numbers[:6]]
                return choices
        
        # 3️⃣ 匹配 "Answer: X" 格式
        answer_pattern = r'Answer\s*[:：]\s*([1-4])'
        answer_matches = re.findall(answer_pattern, response)
        if len(answer_matches) >= 6:
            choices = [int(m) for m in answer_matches[:6]]
            return choices
        
        # 4️⃣ 匹配 "1:1 2:2 3:3..." 格式
        pattern1 = r'(\d+)\s*[:：]\s*([1-4])'
        matches = re.findall(pattern1, response)
        if matches and len(matches) >= 6:
            matches.sort(key=lambda x: int(x[0]))
            choices = [int(m[1]) for m in matches[:6]]
            return choices
        
        # 5️⃣ 尝试匹配逗号分隔的格式
        comma_pattern = r'(?:^|[^\d])([1-4])(?:[,，\s]|$)'
        comma_matches = re.findall(comma_pattern, response)
        if len(comma_matches) >= 6:
            choices = [int(m) for m in comma_matches[:6]]
            return choices
        
        # ❌ 解析失败
        print(f"  ⚠️  WARNING: 无法解析模型响应: {response[:150]}")
        return [0, 0, 0, 0, 0, 0]
    
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
