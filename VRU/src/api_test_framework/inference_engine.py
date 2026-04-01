"""
推理引擎模块
负责调用各类模型进行推理
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple
from .config import MultiModelConfig
from .model_factory import ModelFactory


class InferenceEngine:
    """统一推理引擎"""

    def __init__(self, config: MultiModelConfig, factory: ModelFactory, prompt: str):
        self.config = config
        self.factory = factory
        self.prompt = prompt

    def infer(self, model_name: str, video_frames: np.ndarray) -> Tuple[str, Optional[str]]:
        """推理（返回：text, error）"""
        model_cfg = self.config.models[model_name]

        if model_cfg.type == "api":
            return self._infer_api(model_name, video_frames)
        else:
            return self._infer_local(model_name, video_frames)

    def _infer_api(self, model_name: str, video_frames: np.ndarray) -> Tuple[str, Optional[str]]:
        """推理 API 模型"""
        try:
            if model_name == "qwen_vl":
                return self._infer_qwen_api(video_frames)
            elif model_name == "gpt_4o_mini":
                return self._infer_gpt(video_frames)
            elif model_name == "gemini_2_0_flash":
                return self._infer_gemini(video_frames)
            else:
                return "", f"Unsupported API model: {model_name}"
        except Exception as e:
            return "", str(e)

    def _infer_qwen_api(self, video_frames: np.ndarray) -> Tuple[str, Optional[str]]:
        """Qwen API（DashScope）"""
        from openai import OpenAI
        import base64
        from io import BytesIO
        from PIL import Image

        api_key = self.config.api_keys["qwen"]
        if not api_key:
            return "", "Missing ALI_INTERNATIONAL_KEY"

        try:
            client = OpenAI(
                api_key=api_key,
                base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
            )

            frame_content = []
            for frame in video_frames[:4]:
                img = Image.fromarray(frame)
                buf = BytesIO()
                img.save(buf, format="JPEG")
                b64 = base64.b64encode(buf.getvalue()).decode()
                frame_content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{b64}"
                    }
                })

            content = [{"type": "text", "text": self.prompt}] + frame_content

            resp = client.chat.completions.create(
                model="qwen-vl-max",
                messages=[{"role": "user", "content": content}],
                timeout=60
            )
            return resp.choices[0].message.content, None
        except Exception as e:
            error_msg = str(e)
            print(f"⚠️ Qwen API 错误详情: {error_msg[:200]}")
            return "", error_msg

    def _infer_gpt(self, video_frames: np.ndarray) -> Tuple[str, Optional[str]]:
        """GPT-4o-mini API"""
        from openai import OpenAI
        import base64
        from io import BytesIO
        from PIL import Image

        api_key = self.config.api_keys["openai"]
        if not api_key:
            return "", "Missing OPENAI_API_KEY"

        try:
            client = OpenAI(api_key=api_key)

            frame_b64_list = []
            for frame in video_frames[:4]:
                img = Image.fromarray(frame)
                buf = BytesIO()
                img.save(buf, format="PNG")
                b64 = base64.b64encode(buf.getvalue()).decode()
                frame_b64_list.append(b64)

            content = [{"type": "text", "text": self.prompt}]
            for b64 in frame_b64_list:
                content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{b64}"}
                })

            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": content}],
                timeout=60
            )
            return resp.choices[0].message.content, None
        except Exception as e:
            error_msg = str(e)
            if "deactivated" in error_msg.lower():
                return "", "OpenAI account deactivated or invalid API key"
            print(f"⚠️ GPT API 错误详情: {error_msg[:200]}")
            return "", error_msg

    def _infer_gemini(self, video_frames: np.ndarray) -> Tuple[str, Optional[str]]:
        """Gemini API"""
        import google.generativeai as genai
        from PIL import Image
        import time

        api_key = self.config.api_keys.get("gemini")
        if not api_key:
            return "", "Missing GEMINI_API_KEY"

        try:
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel("models/gemini-2.0-flash")

            selected_frames = video_frames[::max(1, len(video_frames) // 4)][:8]
            images = [Image.fromarray(frame) for frame in selected_frames]

            content = [self.prompt] + images

            safety_settings = [
                {
                    "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                    "threshold": "BLOCK_NONE",
                },
                {
                    "category": "HARM_CATEGORY_HARASSMENT",
                    "threshold": "BLOCK_NONE",
                },
                {
                    "category": "HARM_CATEGORY_HATE_SPEECH",
                    "threshold": "BLOCK_NONE",
                },
                {
                    "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                    "threshold": "BLOCK_NONE",
                },
            ]

            max_retries = 3
            last_error = None

            for attempt in range(max_retries):
                try:
                    resp = model.generate_content(
                        content,
                        generation_config=genai.types.GenerationConfig(
                            temperature=0.2,
                            max_output_tokens=500,
                            top_p=0.8,
                            top_k=20,
                        ),
                        safety_settings=safety_settings,
                        request_options={"timeout": 120}
                    )

                    if resp.text and resp.text.strip():
                        return resp.text, None

                    last_error = "Empty response"
                    time.sleep(2 ** attempt)

                except Exception as e:
                    last_error = str(e)
                    if attempt < max_retries - 1:
                        time.sleep(2 ** attempt)
                        continue

            return "", f"Gemini failed: {last_error}"

        except Exception as e:
            return "", f"Gemini initialization error: {str(e)}"

    def _infer_local(self, model_name: str, video_frames: np.ndarray) -> Tuple[str, Optional[str]]:
        """推理本地模型"""
        model_bundle = self.factory.load_model(model_name)
        if model_bundle is None:
            return "", "Failed to load model"

        try:
            if model_name in ["internvl_1b", "internvl_2b", "internvl_4b", "internvl_8b", "internvl_best"]:
                return self._infer_internvl(model_bundle, video_frames)
            elif model_name == "qwen2_vl_7b":
                return self._infer_qwen2_vl_local(model_bundle, video_frames)
            elif model_name == "llava_next_video":
                return self._infer_llava_next(model_bundle, video_frames)
            elif model_name == "llava_onevision":
                return self._infer_llava_onevision_local(model_bundle, video_frames)
        except Exception as e:
            return "", str(e)
        # finally:
        #     if model_name in ["internvl_1b", "internvl_2b", "internvl_4b", "internvl_8b", "internvl_best", "qwen2_vl_7b", "llava_next_video"]:
        #         self.factory.unload_model(model_name)

    # def _infer_internvl(self, model_bundle: dict, video_frames: np.ndarray) -> Tuple[str, Optional[str]]:
    #     """InternVL2.5-4B推理 - 官方chat接口"""
    #     from PIL import Image

    #     try:
    #         model = model_bundle["model"]
            
    #         # 确保是 numpy 数组
    #         if not isinstance(video_frames, np.ndarray):
    #             video_frames = np.array(video_frames)
            
    #         # 提取前4帧
    #         if len(video_frames.shape) == 4:
    #             frames = video_frames[:min(4, len(video_frames))]
    #         else:
    #             frames = np.array([video_frames])
            
    #         # 转换为PIL图像
    #         images = []
    #         for frame in frames:
    #             # 确保是 uint8
    #             if frame.dtype != np.uint8:
    #                 frame = (frame * 255).astype(np.uint8) if frame.max() <= 1 else frame.astype(np.uint8)
    #             images.append(Image.fromarray(frame))
            
    #         print(f"📌 InternVL: 处理 {len(images)} 帧", flush=True)
            
    #         # 使用官方 chat 方法（不用 processor）
    #         with torch.no_grad():
    #             response = model.chat(
    #                 image=images,
    #                 text=self.prompt,
    #                 do_sample=False,
    #                 max_new_tokens=512
    #             )
            
    #         return response, None
            
    #     except Exception as e:
    #         import traceback
    #         return "", f"InternVL推理失败: {str(e)}\n{traceback.format_exc()}"

    def _infer_internvl(self, model_bundle: dict, video_frames: np.ndarray) -> Tuple[str, Optional[str]]:
        """InternVL 推理 - 自适应调用 model.chat，兼容不同实现签名"""
        import numpy as _np
        from PIL import Image
        import torch
        import inspect
        import traceback

        try:
            model = model_bundle["model"]
            processor = model_bundle.get("processor", None)
            tokenizer = model_bundle.get("tokenizer", None)

            # 统一为 numpy array -> 帧列表 (H,W,C) uint8
            if not isinstance(video_frames, _np.ndarray):
                video_frames = _np.array(video_frames)

            if video_frames.ndim == 3:  # single frame HWC
                frames = [video_frames]
            elif video_frames.ndim == 4:  # (N,H,W,C)
                frames = list(video_frames[: min(4, video_frames.shape[0])])  # 使用最多前4帧
            else:
                return "", "InternVL推理失败: 输入帧维度不支持"

            images = []
            for f in frames:
                if isinstance(f, _np.ndarray):
                    if f.dtype != _np.uint8:
                        f = (f * 255).astype(_np.uint8) if f.max() <= 1 else f.astype(_np.uint8)
                    img = Image.fromarray(f)
                else:
                    img = f
                # 确保 RGB 模式，避免灰度/RGBA 触发断言
                if hasattr(img, "convert"):
                    img = img.convert("RGB")
                images.append(img)

            # 避免某些实现对多帧有断言限制，优先使用1帧
            if len(images) > 1:
                images_single = [images[0]]
            else:
                images_single = images

            print(f"DEBUG: tokenizer={type(tokenizer)}, processor={type(processor)}, images_count={len(images)}", flush=True)

            gen_cfg = {"max_new_tokens": 512, "do_sample": False}

            last_exc = None
            # 先通过inspect拿签名，判断接收哪些参数
            try:
                sig = inspect.signature(model.chat)
                params = list(sig.parameters.keys())
            except Exception:
                params = []

            # 尝试几种常见调用方式（有些实现需要 processor, 有些不）
            try_variants = []

            # 优先：InternVL3 官方 chat 风格（使用 pixel_values）
            if tokenizer is not None:
                # 将 PIL images 转为 tensor (官方示例格式)
                try:
                    import torchvision.transforms as T
                    from torchvision.transforms.functional import InterpolationMode
                    MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
                    transform = T.Compose([
                        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
                        T.Resize((448, 448), interpolation=InterpolationMode.BICUBIC),
                        T.ToTensor(),
                        T.Normalize(mean=MEAN, std=STD)
                    ])
                    pixel_values = torch.stack([transform(img) for img in images_single]).to(torch.bfloat16).to(self.config.device)
                    
                    # 官方标准调用: model.chat(tokenizer, pixel_values, question, generation_config)
                    try_variants.append(lambda: model.chat(tokenizer, pixel_values, self.prompt, gen_cfg))
                    try_variants.append(lambda: model.chat(tokenizer, pixel_values, question=self.prompt, generation_config=gen_cfg))
                except Exception as e:
                    print(f"Warning: pixel_values transform failed: {e}", flush=True)

            # 常见： model.chat(processor, images, question, generation_config=...)
            if processor is not None:
                try_variants.append(lambda: model.chat(processor, images, self.prompt, generation_config=gen_cfg))
                try_variants.append(lambda: model.chat(processor, images, question=self.prompt, generation_config=gen_cfg))
                try_variants.append(lambda: model.chat(processor=processor, images=images, question=self.prompt, generation_config=gen_cfg))

            # 常见2： model.chat(images, question, generation_config=...)
            try_variants.append(lambda: model.chat(images_single, self.prompt, generation_config=gen_cfg))
            try_variants.append(lambda: model.chat(images=images_single, question=self.prompt, generation_config=gen_cfg))
            try_variants.append(lambda: model.chat(images=images_single, text=self.prompt, generation_config=gen_cfg))
            try_variants.append(lambda: model.chat(images_single, text=self.prompt, generation_config=gen_cfg))

            # InternVL 官方实现常见：model.chat(image=[...], text=...)
            try_variants.append(lambda: model.chat(image=images_single, text=self.prompt, do_sample=False, max_new_tokens=512))

            # 若上述均失败，尝试降级：用 processor(inputs) -> model.generate / model.generate_text
            def fallback_generate():
                if processor is None:
                    raise RuntimeError("no processor for fallback")
                inputs = processor(images=images_single, text=self.prompt, return_tensors="pt")
                # 将 tensors 移到设备
                def _to_device(obj, device):
                    if isinstance(obj, dict):
                        return {k: _to_device(v, device) for k, v in obj.items()}
                    if hasattr(obj, "to"):
                        return obj.to(device)
                    return obj
                inputs = _to_device(inputs, self.config.device)
                if hasattr(model, "generate"):
                    out = model.generate(**inputs, max_new_tokens=512)
                    # decode via processor if有tokenizer
                    if hasattr(processor, "decode"):
                        return processor.decode(out[0], skip_special_tokens=True)
                    return str(out)
                if hasattr(model, "generate_text"):
                    return model.generate_text(**inputs, max_new_tokens=512)
                raise RuntimeError("no generate interface available")

            try_variants.append(fallback_generate)

            # 依次尝试
            for idx, fn in enumerate(try_variants):
                try:
                    res = fn()
                    # 若返回不是str，尝试转换
                    if res is None:
                        continue
                    if not isinstance(res, str):
                        try:
                            res = str(res)
                        except Exception:
                            pass
                    return res, None
                except Exception as e:
                    last_exc = e

            # 全部失败，返回最后异常和 traceback
            # 提供更明确的异常类型与消息
            msg = ""
            try:
                msg = f"{type(last_exc).__name__}: {str(last_exc)}"
            except Exception:
                msg = "Unknown error"
            return "", f"InternVL推理失败: {msg}\n{traceback.format_exc()}"

        except Exception as e:
            return "", f"InternVL推理失败: {str(e)}\n{traceback.format_exc()}"

    def _infer_qwen2_vl_local(self, model_bundle: dict, video_frames: np.ndarray) -> Tuple[str, Optional[str]]:
        """Qwen2-VL-7B本地推理"""
        from PIL import Image

        model = model_bundle["model"]
        processor = model_bundle["processor"]
        images = [Image.fromarray(frame) for frame in video_frames]

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": img} for img in images
                ] + [{"type": "text", "text": self.prompt}]
            }
        ]

        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=text, images=images, return_tensors="pt").to("cuda")

        with torch.no_grad():
            output_ids = model.generate(**inputs, max_new_tokens=500, do_sample=False)

        response = processor.decode(output_ids[0], skip_special_tokens=True)
        return response, None

    def _infer_llava_next(self, model_bundle: dict, video_frames: np.ndarray) -> Tuple[str, Optional[str]]:
        """LLaVA-NeXT-Video推理"""
        model = model_bundle["model"]
        processor = model_bundle["processor"]
        video_tensor = torch.from_numpy(video_frames).permute(0, 3, 1, 2).float()

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": self.prompt},
                    {"type": "video"},
                ],
            }
        ]

        prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = processor(
            text=prompt,
            videos=video_tensor,
            padding=True,
            return_tensors="pt"
        ).to(model.device)

        with torch.inference_mode():
            output_ids = model.generate(**inputs, max_new_tokens=500, do_sample=False)

        response = processor.decode(output_ids[0][2:], skip_special_tokens=True)
        return response, None

    def _infer_llava_onevision_local(self, model_bundle: dict, video_frames: np.ndarray) -> Tuple[str, Optional[str]]:
        """LLaVA-OneVision推理"""
        from PIL import Image

        model = model_bundle["model"]
        processor = model_bundle["processor"]
        images = [Image.fromarray(frame) for frame in video_frames]

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": self.prompt},
                    *[{"type": "image"} for _ in images]
                ],
            }
        ]

        prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = processor(text=prompt, images=images, return_tensors="pt").to("cuda")

        with torch.inference_mode():
            output_ids = model.generate(**inputs, max_new_tokens=1000, do_sample=False)

        response = processor.decode(output_ids[0][2:], skip_special_tokens=True)
        return response, None
