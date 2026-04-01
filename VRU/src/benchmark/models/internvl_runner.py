import torch
import gc
import re
import math
# import logging
from PIL import Image
import numpy as np
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
import bitsandbytes

# logger = logging.getLogger(__name__)

# ImageNet 标准化参数
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transform(input_size):
    """构建图像预处理transform"""
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform


def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    """找到最接近的宽高比"""
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    """动态预处理图像（InternVL的核心方法）"""
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # 计算可能的宽高比
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # 找到最佳宽高比
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # 计算目标宽度和高度
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # 调整图像大小
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # 分割图像
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    
    return processed_images


class InternVLRunner:
    """InternVL 系列通用 Runner - 参考VRU-Accident项目实现
    
    支持的模型：
    - internvl2-2b, internvl2-4b, internvl2-8b
    - internvl2.5-1b, internvl2.5-2b, internvl2.5-4b, internvl2.5-8b  
    - internvl3-1b, internvl3-2b, internvl3-4b, internvl3-8b
    """
    
    def __init__(self, model_name):
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._load_model()
        # print('info:', type(self.model), '\n')
        # print(dir(self.model))
    
    def _load_model(self):
        """加载 InternVL 系列模型"""
        try:
            from transformers import AutoModel, AutoTokenizer
            
            print(f"  📥 加载 {self.model_name}...")
            
            # 模型名称映射
            model_map = {
                'internvl2-2b': 'OpenGVLab/InternVL2-2B',
                'internvl2-4b': 'OpenGVLab/InternVL2-4B',
                'internvl2-8b': 'OpenGVLab/InternVL2-8B',
                'internvl2.5-1b': 'OpenGVLab/InternVL2_5-1B',
                'internvl2.5-2b': 'OpenGVLab/InternVL2_5-2B',
                'internvl2.5-4b': 'OpenGVLab/InternVL2_5-4B',
                'internvl2.5-8b': 'OpenGVLab/InternVL2_5-8B',
                'internvl3-1b': 'OpenGVLab/InternVL3-1B',
                'internvl3-2b': 'OpenGVLab/InternVL3-2B',
                'internvl3-8b': 'OpenGVLab/InternVL3-8B',
                'internvl3-9b': 'OpenGVLab/InternVL3-9B',
                'internvl3_5-1b': 'OpenGVLab/InternVL3_5-1B',
                'internvl3_5-2b': 'OpenGVLab/InternVL3_5-2B',
                'internvl3_5-8b': 'OpenGVLab/InternVL3_5-8B',
            }
            
            model_id = model_map.get(self.model_name)
            if not model_id:
                raise ValueError(f"不支持的模型: {self.model_name}")
            
            # 加载tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_id, trust_remote_code=True, use_fast=False
            )
            
            # 加载模型 - 根据模型版本选择不同的加载方式
            # InternVL3 系列需要特殊处理
            if 'internvl3' in self.model_name.lower():
                print(f"  ℹ️  InternVL3 检测到，尝试修复 language_model.generate 问题")
                
                # 先尝试标准加载
                self.model = AutoModel.from_pretrained(
                    model_id,
                    torch_dtype=torch.bfloat16,
                    low_cpu_mem_usage=True,
                    use_flash_attn=False,
                    trust_remote_code=True,
                    device_map="auto",
                    load_in_8bit=True
                ).eval()
                
                # 检查并修复 language_model
                if hasattr(self.model, 'language_model'):
                    lm = self.model.language_model
                    print(f"  ℹ️  language_model 类型: {type(lm).__name__}")
                    
                    # 如果 language_model 没有 generate，尝试从父类继承
                    if not hasattr(lm, 'generate') or not callable(getattr(lm, 'generate', None)):
                        print(f"  ⚠️  修复 language_model.generate 缺失")
                        # 直接从 PreTrainedModel 和 GenerationMixin 继承所有方法
                        from transformers.generation.utils import GenerationMixin
                        from transformers.modeling_utils import PreTrainedModel
                        
                        # 动态添加 GenerationMixin 到类的基类
                        if GenerationMixin not in lm.__class__.__bases__:
                            lm.__class__.__bases__ = lm.__class__.__bases__ + (GenerationMixin,)
                            print(f"  ✓ 已添加 GenerationMixin 到 {type(lm).__name__}")
            else:
                # InternVL2/2.5 使用原有的 8-bit 量化加载方式
                self.model = AutoModel.from_pretrained(
                    model_id,
                    torch_dtype=torch.bfloat16,
                    low_cpu_mem_usage=True,
                    use_flash_attn=False,
                    trust_remote_code=True,
                    device_map="auto",
                    load_in_8bit=True
                ).eval()
            
            # 设置必要的 token IDs（InternVL 必须设置）
            if hasattr(self.model, 'img_context_token_id'):
                # 从 tokenizer 获取 <IMG_CONTEXT> token 的 ID
                img_context_token_id = self.tokenizer.convert_tokens_to_ids('<IMG_CONTEXT>')
                self.model.img_context_token_id = img_context_token_id
                print(f"  ℹ️  设置 img_context_token_id = {img_context_token_id}")
            
            print(f"  ✓ {self.model_name} 加载成功")
            
        except Exception as e:
            print(f"  ❌ 加载失败: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def predict(self, video_number, prompt, video_frames, num_frames=10, num_options=3):
        """推理选择题 - 参考VRU-Accident实现
        
        Args:
            video_number: 视频编号
            prompt: 完整的prompt
            video_frames: numpy数组 (num_frames, H, W, 3)
            num_frames: 要使用的帧数（如果为None，自动根据加载的帧数调整，最多8帧）
            num_options: 每题的选项数（3, 4, 或 5）
        
        Returns:
            choices: 选项序号列表
        """
        if self.model is None:
            print(f"  ❌ 模型未加载")
            return [0] * 6
        
        try:
            # 0. 动态控制 token 预算（估算字符/4 作为 token）
            def _estimate_tokens_from_chars(chars: int) -> int:
                return max(1, math.ceil(chars / 4))

            def _estimate_tokens(text: str) -> int:
                return _estimate_tokens_from_chars(len(text))

            max_context_tokens = 3800  # 为模型上下文预留安全余量（4096 以内）
            prompt_tokens = _estimate_tokens(prompt)

            # 1. 使用 dynamic_preprocess 处理视频帧（VRU-Accident方式）
            num_frames_total = len(video_frames)
            
            # 根据num_frames参数决定采样帧数（不限制为8帧，让不同的num_frames产生不同结果）
            if num_frames is None:
                # 如果未指定num_frames，则使用全部加载的帧
                num_segments = num_frames_total
            else:
                # 如果指定了num_frames，则采样到该值（但不超过实际加载的帧数）
                num_segments = min(num_frames, num_frames_total)
            
            # 如果加载的帧数少于等于所需帧数，直接使用全部帧，否则均匀采样
            if num_frames_total <= num_segments:
                frame_indices = np.arange(num_frames_total, dtype=int)
            else:
                frame_indices = np.linspace(0, num_frames_total - 1, num_segments, dtype=int)
            
            pixel_values_list = []
            num_patches_list = []
            transform = build_transform(input_size=448)
            
            # 基于帧前缀的开销调整帧数，避免文本+帧前缀超出上下文
            per_frame_prefix_tokens = _estimate_tokens_from_chars(len("Frame000: <image>\n"))
            num_segments_after_check = len(frame_indices)
            while num_segments_after_check > 1 and (prompt_tokens + num_segments_after_check * per_frame_prefix_tokens) > max_context_tokens:
                num_segments_after_check -= 1
                if num_frames_total <= 8:
                    frame_indices = frame_indices[:num_segments_after_check]
                else:
                    frame_indices = np.linspace(0, num_frames_total - 1, num_segments_after_check, dtype=int)

            if (prompt_tokens + num_segments_after_check * per_frame_prefix_tokens) > max_context_tokens:
                print(f"  ⚠️ prompt过长，文本已接近上下文上限（估算 {prompt_tokens} tokens），仅保留 {num_segments_after_check} 帧")
            
            for frame_idx in frame_indices:
                frame = video_frames[frame_idx]
                img = Image.fromarray(frame).convert('RGB')
                
                # 使用 dynamic_preprocess（VRU-Accident的关键方法）
                img_tiles = dynamic_preprocess(img, image_size=448, use_thumbnail=True, max_num=1)
                pixel_values = [transform(tile) for tile in img_tiles]
                pixel_values = torch.stack(pixel_values)
                num_patches_list.append(pixel_values.shape[0])
                pixel_values_list.append(pixel_values)
            
            pixel_values = torch.cat(pixel_values_list)
            pixel_values = pixel_values.to(torch.bfloat16).to(self.device)
            
            # 2. 构造conversation（VRU-Accident格式）
            video_prefix = ''.join([f'Frame{i+1}: <image>\n' for i in range(len(num_patches_list))])
            conversation = video_prefix + prompt
            
            # 3. 调用模型chat方法（VRU-Accident方式）
            # 根据剩余额度调整生成长度
            used_tokens = prompt_tokens + len(num_patches_list) * per_frame_prefix_tokens
            available_for_gen = max(64, max_context_tokens - used_tokens)
            generation_config = {
                'max_new_tokens': min(64, max(16, available_for_gen // 2)),
                'do_sample': False
            }
            
            # print(f"  ℹ️ 生成配置: max_new_tokens={actual_max_tokens}, 使用帧数={num_segments}")
            
            with torch.no_grad():
                response, history = self.model.chat(
                    self.tokenizer,
                    pixel_values,
                    conversation,
                    generation_config,
                    num_patches_list=num_patches_list,
                    history=None,
                    return_history=True
                )
            
            # 4. 从prompt中提取题目数量
            q_matches = re.findall(r'Q(\d+):', prompt)
            num_questions = len(q_matches) if q_matches else 6
            
            # 5. 解析选项
            response_preview = response if len(response) < 150 else response[:150] + "..."
            print(f"  📝 原始响应: {response_preview}")
            choices = self._parse_choices(response, num_questions, num_options=num_options)
            
            return choices
            
        except Exception as e:
            print(f"  ❌ 推理失败: {e}")
            import traceback
            traceback.print_exc()
            return [0] * 6
    
    def _parse_choices(self, response: str, num_questions: int = 6, num_options: int = 3) -> list:
        """解析选项序号
        
        Args:
            response: 模型响应文本
            num_questions: 问题数量
            num_options: 每题的选项数（3, 4, 或 5）
        """
        response = response.strip()
        
        # 策略1: 匹配 "A1: 1\nA2: 2..." 格式
        pattern1 = r'[Aa](\d+)\s*[:\：]\s*(\d+)'
        matches = re.findall(pattern1, response)
        
        if matches:
            # 按问题号排序
            matches.sort(key=lambda x: int(x[0]))
            extracted = []
            for q_num, choice_str in matches:
                choice = int(choice_str)
                if 1 <= choice <= num_options:
                    extracted.append(choice)
            
            if len(extracted) >= num_questions:
                result = extracted[:num_questions]
                print(f"  ✓ 策略1成功 (A1:1格式): {result}")
                return result
            elif len(extracted) > 0:
                result = extracted + [0] * (num_questions - len(extracted))
                print(f"  ⚠️ 策略1部分成功 (找到{len(extracted)}/{num_questions}): {result}")
                return result
        
        # 策略2: 提取所有在范围内[1,num_options]的数字
        numbers = re.findall(r'\d', response)
        extracted = [int(n) for n in numbers if 1 <= int(n) <= num_options]
        
        if len(extracted) >= num_questions:
            result = extracted[:num_questions]
            print(f"  ✓ 策略2成功 (数字提取): {result}")
            return result
        elif len(extracted) > 0:
            result = extracted + [0] * (num_questions - len(extracted))
            print(f"  ⚠️ 策略2部分成功 (找到{len(extracted)}/{num_questions}): {result}")
            return result
        
        # 所有策略失败
        print(f"  ❌ 解析失败 - 响应: {response[:100]}...")
        return [0] * num_questions
    
    # def predict_single(self, video_number, prompt, video_frames, question_idx=1):
    #     """推理单个问题 - 分别处理每个问题以获得更好的效果
        
    #     Args:
    #         video_number: 视频编号
    #         prompt: 单个问题的提示词
    #         video_frames: numpy数组 (num_frames, H, W, 3)
    #         question_idx: 问题索引（用于日志）
        
    #     Returns:
    #         choice: 选项序号（1-4）
    #     """
    #     if self.model is None:
    #         print(f"  ❌ 模型未加载")
    #         return 0
        
    #     try:
    #         # 单问题模式：更多token用于思考
    #         def _estimate_tokens_from_chars(chars: int) -> int:
    #             return max(1, math.ceil(chars / 4))

    #         def _estimate_tokens(text: str) -> int:
    #             return _estimate_tokens_from_chars(len(text))

    #         max_context_tokens = 3900  # 给单问题更多空间
    #         prompt_tokens = _estimate_tokens(prompt)

    #         # 处理视频帧
    #         num_frames_total = len(video_frames)
    #         num_segments = min(24, num_frames_total)

    #         # 基于帧前缀的开销调整帧数
    #         per_frame_prefix_tokens = _estimate_tokens_from_chars(len("Frame000: <image>\n"))
    #         while num_segments > 1 and (prompt_tokens + num_segments * per_frame_prefix_tokens) > max_context_tokens:
    #             num_segments -= 1

    #         # 均匀采样帧索引
    #         frame_indices = np.linspace(0, num_frames_total - 1, num_segments, dtype=int)
            
    #         pixel_values_list = []
    #         num_patches_list = []
    #         transform = build_transform(input_size=448)
            
    #         for frame_idx in frame_indices:
    #             frame = video_frames[frame_idx]
    #             img = Image.fromarray(frame).convert('RGB')
                
    #             # 使用 dynamic_preprocess
    #             img_tiles = dynamic_preprocess(img, image_size=448, use_thumbnail=True, max_num=1)
    #             pixel_values = [transform(tile) for tile in img_tiles]
    #             pixel_values = torch.stack(pixel_values)
    #             num_patches_list.append(pixel_values.shape[0])
    #             pixel_values_list.append(pixel_values)
            
    #         pixel_values = torch.cat(pixel_values_list)
    #         pixel_values = pixel_values.to(torch.bfloat16).to(self.device)
            
    #         # 构造conversation
    #         video_prefix = ''.join([f'Frame{i+1}: <image>\n' for i in range(len(num_patches_list))])
    #         conversation = video_prefix + prompt
            
    #         # 单问题模式分配更多token給生成
    #         used_tokens = prompt_tokens + len(num_patches_list) * per_frame_prefix_tokens
    #         available_for_gen = max_context_tokens - used_tokens
    #         if available_for_gen < 0:
    #             available_for_gen = 0
    #         generation_config = {
    #             'max_new_tokens': min(128, max(32, available_for_gen // 2)),  # 比批量模式更多的生成token
    #             'do_sample': False
    #         }
            
    #         if 'internvl3' in self.model_name:
    #             with torch.no_grad():
    #                 response, history = self.model.generate(
    #                     self.tokenizer,
    #                     pixel_values,
    #                     conversation,
    #                     generation_config,
    #                 num_patches_list=num_patches_list,
    #                 history=None,
    #                 return_history=True
    #             )
    #         else:
    #             with torch.no_grad():
    #                 response, history = self.model.chat(
    #                     self.tokenizer,
    #                     pixel_values,
    #                     conversation,
    #                     generation_config,
    #                     num_patches_list=num_patches_list,
    #                     history=None,
    #                     return_history=True
    #                 )
            
    #         # 解析单个选项
    #         choice = self._parse_single_choice(response)
            
    #         print(f"  🔍 DEBUG - Q{question_idx} 原始响应: {response[:100]}")
    #         print(f"  🔍 DEBUG - Q{question_idx} 解析结果: {choice}")
    #         print(f"  ✓ {self.model_name} 推理完成: {video_number}/Q{question_idx} (使用{num_segments}帧)")
            
    #         return choice
            
    #     except Exception as e:
    #         print(f"  ❌ 推理失败: {e}")
    #         return 0
    
    def _parse_single_choice(self, response):
        """从响应中解析单个选项（1-4）"""
        # 1️⃣ 寻找 "A: [数字]" 的模式
        pattern = r'A[:\s]+([1-4])'
        match = re.search(pattern, response, re.IGNORECASE)
        if match:
            return int(match.group(1))
        
        # 2️⃣ 直接提取任何单个1-4的数字（word boundaries）
        numbers = re.findall(r'\b[1-4]\b', response)
        if numbers:
            return int(numbers[0])
        
        # 3️⃣ 如果word boundary失败，尝试宽松匹配
        numbers = re.findall(r'[1-4]', response)
        if numbers:
            return int(numbers[0])
        
        print(f"  ⚠️ 无法解析单个选项 - {response[:100]}")
        return 0
    
    def release(self):
        """释放资源"""
        if self.model is not None:
            del self.model
            self.model = None
        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"  ♻️ {self.model_name} 资源已释放")

