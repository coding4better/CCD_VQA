import torch
import gc
import re
from PIL import Image
import numpy as np


class QwenRunner:
    """Qwen2.5-VL 视频理解推理Runner
    
    支持：
    - 本地推理：Qwen2.5-VL-7B-Instruct（推荐）
    """
    
    def __init__(self, model_name, num_frames=4):
        import os
        
        # 在初始化前设置CUDA内存优化
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
        os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
        
        self.model_name = model_name
        self.model = None
        self.processor = None
        self.is_api = model_name.endswith('-api')
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.num_frames = num_frames  # None 表示使用全部帧，数字表示使用指定帧数
        
        if not self.is_api:
            self._load_model()
    
    def _load_model(self):
        """加载 Qwen2.5-VL-7B-Instruct 本地模型"""
        try:
            from transformers import AutoProcessor
            from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLForConditionalGeneration
            import os
            
            print(f"  📥 加载 {self.model_name}...")
            print(f"  🔧 启用内存优化模式...")
            
            # 查找本地模型路径
            huggingface_cache = os.path.expanduser("~/.cache/huggingface/hub")
            
            # 支持多种本地缓存目录格式
            local_model_paths = [
                os.path.join(huggingface_cache, "Qwen2.5-VL-7B-Instruct"),
                os.path.join(huggingface_cache, "models--Qwen--Qwen2.5-VL-7B-Instruct/snapshots"),
                os.path.join(huggingface_cache, "Qwen2.5-VL-7B"),
            ]
            
            model_path = None
            for path in local_model_paths:
                if os.path.exists(path):
                    # 如果是snapshots目录，找到第一个snapshot
                    if "snapshots" in path:
                        snapshots = os.listdir(path)
                        if snapshots:
                            model_path = os.path.join(path, snapshots[0])
                    else:
                        model_path = path
                    
                    if model_path and os.path.exists(os.path.join(model_path, "config.json")):
                        print(f"  ✓ 找到本地模型: {model_path}")
                        break
            
            if not model_path:
                raise FileNotFoundError(
                    f"❌ 未找到本地模型!\n"
                    f"  请检查是否已下载: {huggingface_cache}\n"
                    f"  期望目录: Qwen2.5-VL-7B-Instruct 或 models--Qwen--Qwen2.5-VL-7B-Instruct"
                )
            
            # 使用标准加载方式（不量化），以获得更好的推理质量
            print(f"  💾 以标准精度加载模型（获得最佳推理质量）...")
            
            load_kwargs = {
                "low_cpu_mem_usage": True,
                "device_map": "auto",
                "local_files_only": True,
                "torch_dtype": torch.float16,  # 使用 float16 节省一些显存但保持精度
                "trust_remote_code": True,
            }
            
            # Qwen2.5-VL 用 Qwen2_5_VLForConditionalGeneration 加载以获得 generate() 方法
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_path,
                **load_kwargs
            ).eval()
            
            self.processor = AutoProcessor.from_pretrained(
                model_path, 
                local_files_only=True
            )
            
            # 清理缓存
            gc.collect()
            torch.cuda.empty_cache()
            
            print(f"  ✓ {self.model_name} 加载成功 ({self.device})")
            print(f"  ✓ 显存优化完成，当前占用: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
            
        except ImportError as e:
            print(f"  ❌ 缺失依赖: {e}")
            print(f"  💡 尝试安装: pip install bitsandbytes")
            raise
        except Exception as e:
            print(f"  ❌ 加载失败: {e}")
            raise
    
    def predict(self, video_number, prompt, video_frames, num_options=4, expected_count=6):
        """推理选择题，使用全部视频帧"""
        if self.is_api:
            return self._predict_api(prompt, num_options=num_options, expected_count=expected_count)
        else:
            return self._predict_local(video_number, prompt, video_frames, num_options=num_options, expected_count=expected_count)
    
    def _predict_local(self, video_number, prompt, video_frames, num_options=4, expected_count=6):
        """本地推理，可自定义使用的帧数（内存优化）"""
        if self.model is None:
            print(f"  ❌ 模型未加载")
            return [0] * max(1, expected_count)
        
        try:
            # 清理旧的显存
            torch.cuda.empty_cache()
            gc.collect()
            
            # 根据 num_frames 参数选择帧数
            if self.num_frames is not None:
                # 均匀采样指定数量的帧
                num_frames = min(self.num_frames, len(video_frames))
                if num_frames == len(video_frames):
                    selected_frames = video_frames
                else:
                    indices = np.linspace(0, len(video_frames) - 1, num_frames, dtype=int)
                    selected_frames = [video_frames[i] for i in indices]
                print(f"  📹 使用 {num_frames}/{len(video_frames)} 帧进行推理")
            else:
                # 使用全部帧
                selected_frames = video_frames
                print(f"  📹 使用全部 {len(video_frames)} 帧进行推理")
            
            images = [Image.fromarray(frame) for frame in selected_frames]
            
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt}
                    ] + [{"type": "image", "image": img} for img in images],
                }
            ]
            
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            
            inputs = self.processor(
                text=text,
                images=images,
                return_tensors="pt"
            ).to(self.device)
            
            # 推理前清理
            gc.collect()
            torch.cuda.empty_cache()
            
            with torch.no_grad():
                # Qwen2.5-VL 使用 generate() 方法
                output_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=128,
                    do_sample=False,  # 使用贪心解码
                    use_cache=True,
                )
            
            # 解码输出，跳过输入部分
            response = self.processor.decode(output_ids[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
            
            # DEBUG: 打印模型的原始输出
            print(f"  📝 {self.model_name} 原始响应: {repr(response)}")
            
            choices = self._parse_choices(response, num_options=num_options, expected_count=expected_count)
            print(f"  ✓ {self.model_name} 推理完成: {video_number}, 解析结果: {choices}")
            
            # 推理后清理
            del inputs
            del output_ids
            gc.collect()
            torch.cuda.empty_cache()
            
            return choices
            
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"  ❌ 显存不足，建议:")
                print(f"     1. 减少帧数: num_frames=4 或更小")
                print(f"     2. 使用API版本: model_name='qwen-vl-api'")
                print(f"     3. 清理其他进程的GPU占用")
            else:
                print(f"  ❌ 推理失败: {e}")
            return [0] * max(1, expected_count)
        except Exception as e:
            print(f"  ❌ 推理失败: {e}")
            return [0] * max(1, expected_count)
    
    def _predict_api(self, prompt, num_options=4, expected_count=6):
        """API 推理（Qwen VL API）"""
        try:
            import os
            from openai import OpenAI
            
            api_key = os.getenv("DASHSCOPE_API_KEY")
            if not api_key:
                print("  ❌ 未设置 DASHSCOPE_API_KEY 环境变量")
                return [0] * max(1, expected_count)
            
            client = OpenAI(
                api_key=api_key,
                base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
            )
            
            # 简单文本推理（不含图片）
            resp = client.chat.completions.create(
                model="qwen-vl-max",
                messages=[{"role": "user", "content": prompt}],
                timeout=60
            )
            
            response = resp.choices[0].message.content
            choices = self._parse_choices(response, num_options=num_options, expected_count=expected_count)
            print(f"  ✓ {self.model_name} API 推理完成")
            return choices
            
        except Exception as e:
            print(f"  ❌ API 推理失败: {e}")
            return [1] * max(1, expected_count)
    
    def _parse_choices(self, response: str, num_options: int = 4, expected_count: int = 6) -> list:
        """解析选项序号 - 支持多种格式"""
        if num_options <= 0:
            num_options = 4
        if expected_count <= 0:
            expected_count = 1

        max_opt = str(num_options)
        opt_char_class = f"[1-{max_opt}]"
        choices = []
        
        # 模式1: "Q1:1 Q2:2" 格式（优先级最高）
        pattern1 = rf'[Qq]\s*(\d+)\s*[：:]\s*({opt_char_class})'
        matches = re.findall(pattern1, response)
        
        if matches and len(matches) >= expected_count:
            matches.sort(key=lambda x: int(x[0]))
            choices = [int(m[1]) for m in matches[:expected_count]]
            return choices
        
        # 模式2: "题1:1" 中文格式
        pattern2 = rf'题\s*(\d+)\s*[：:]\s*({opt_char_class})'
        matches2 = re.findall(pattern2, response)
        if matches2 and len(matches2) >= expected_count:
            matches2.sort(key=lambda x: int(x[0]))
            choices = [int(m[1]) for m in matches2[:expected_count]]
            return choices
        
        # 模式3: 简单数字序列，但要更智能地选择前6个
        # 优先选择那些看起来像答案的（即紧跟在问号或数字后面的）
        pattern3 = opt_char_class
        all_numbers = re.findall(pattern3, response)
        
        # 尝试找包含"ANSWERS"或答案标记之后的数字
        if 'ANSWERS' in response.upper():
            answer_section = response[response.upper().rfind('ANSWERS'):]
            numbers = re.findall(pattern3, answer_section)
            if len(numbers) >= expected_count:
                choices = [int(n) for n in numbers[:expected_count]]
                return choices
        
        # 最后的备选：只取最后6个数字
        if len(all_numbers) >= expected_count:
            choices = [int(n) for n in all_numbers[-expected_count:]]
            return choices
        
        # 如果完全找不到，发出警告
        if not choices:
            print(f"  ⚠️ 警告：无法从响应解析选项。响应: {response[-200:]}")
            choices = [1] * expected_count
        
        # 过滤和填充
        choices = [c if 1 <= c <= num_options else 1 for c in choices]
        while len(choices) < expected_count:
            choices.append(1)
        
        return choices[:expected_count]
    
    def release(self):
        """释放资源"""
        if self.model is not None:
            del self.model
            self.model = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"  ♻️ {self.model_name} 资源已释放")
