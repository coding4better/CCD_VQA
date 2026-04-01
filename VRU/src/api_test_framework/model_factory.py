"""
模型工厂模块
负责加载和管理各类模型
"""

import gc
import torch
from typing import Optional, Dict, Any
from .config import MultiModelConfig


class ModelFactory:
    """模型加载工厂"""

    def __init__(self, config: MultiModelConfig):
        self.config = config
        self.loaded_models = {}  # 缓存已加载模型

    def load_model(self, model_name: str):
        """加载指定模型（带缓存）"""
        if model_name in self.loaded_models:
            return self.loaded_models[model_name]

        model_cfg = self.config.models[model_name]

        if model_cfg.type == "api":
            # API 模型无需加载，返回 None
            return None

        print(f"📥 加载模型: {model_name} ({model_cfg.size})...")

        try:
            if model_name == "internvl_1b":
                return self._load_internvl("OpenGVLab/InternVL3-1B")
            elif model_name == "internvl_2b":
                return self._load_internvl("OpenGVLab/InternVL3-2B")
            elif model_name == "internvl_4b":
                return self._load_internvl("OpenGVLab/InternVL3-4B")
            elif model_name == "internvl_8b":
                return self._load_internvl("OpenGVLab/InternVL3-8B")
            elif model_name == "internvl_best":
                # 粗略按显存选择：>=14GB 选 8B，>=6GB 选 4B，否则 2B
                try:
                    if torch.cuda.is_available():
                        props = torch.cuda.get_device_properties(0)
                        total_gb = props.total_memory / (1024**3)
                        if total_gb >= 14:
                            return self._load_internvl("OpenGVLab/InternVL3-8B")
                        elif total_gb >= 6:
                            return self._load_internvl("OpenGVLab/InternVL3-4B")
                except Exception:
                    pass
                return self._load_internvl("OpenGVLab/InternVL3-2B")
            elif model_name == "qwen2_vl_7b":
                return self._load_qwen2_vl()
            elif model_name == "llava_next_video":
                return self._load_llava_next_video()
            elif model_name == "llava_onevision":
                return self._load_llava_onevision()
            else:
                raise ValueError(f"未知模型: {model_name}")
        except Exception as e:
            # Print full traceback to logs for easier debugging
            try:
                import traceback

                print("❌ 加载失败:", e)
                traceback.print_exc()
            except Exception:
                # fallback minimal message
                print(f"❌ 加载失败: {e}")
            return None

    def _load_internvl(self, model_id: str = "OpenGVLab/InternVL3-4B"):
        """加载 InternVL3（官方用法）

        - 使用 AutoModel + AutoTokenizer（trust_remote_code=True）
        - 官方 chat 接口: model.chat(tokenizer, pixel_values, question, generation_config)
        - 附带可选 AutoProcessor 以便在必要时回退到 generate 路径
        """
        import torch
        from transformers import AutoModel, AutoProcessor, AutoTokenizer

        model = AutoModel.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            device_map="auto",
        ).eval()

        tokenizer = AutoTokenizer.from_pretrained(
            model_id, trust_remote_code=True, use_fast=False
        )

        try:
            processor = AutoProcessor.from_pretrained(
                model_id, trust_remote_code=True
            )
        except Exception:
            processor = None

        return {"model": model, "tokenizer": tokenizer, "processor": processor, "type": "internvl"}

    # def _load_internvl(self):
        # """加载 InternVL2.5-4B - 修正版"""
        # from transformers import AutoModel, AutoProcessor

        # model = AutoModel.from_pretrained(
        #     "OpenGVLab/InternVL2_5-4B",
        #     torch_dtype=torch.bfloat16,
        #     low_cpu_mem_usage=True,
        #     trust_remote_code=True,
        # ).cuda().eval()
        
        # # 使用 AutoProcessor 而不是 AutoTokenizer
        # processor = AutoProcessor.from_pretrained(
        #     "OpenGVLab/InternVL2_5-4B", 
        #     trust_remote_code=True
        # )
        
        # return {"model": model, "processor": processor, "type": "internvl"}   

    def _load_qwen2_vl(self):
        """加载 Qwen2-VL-7B"""
        from transformers import (
            Qwen2VLForConditionalGeneration,
            AutoTokenizer,
            AutoProcessor,
        )
        from transformers import BitsAndBytesConfig

        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )

        model = Qwen2VLForConditionalGeneration.from_pretrained(
            "Qwen/Qwen2-VL-7B-Instruct",
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            device_map="auto",
            quantization_config=quantization_config
        )
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")
        processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")
        return {
            "model": model,
            "tokenizer": tokenizer,
            "processor": processor,
            "type": "qwen2_vl",
        }

    def _load_llava_next_video(self):
        """加载 LLaVA-NeXT-Video-7B"""
        from transformers import (
            LlavaNextVideoProcessor,
            LlavaNextVideoForConditionalGeneration,
        )

        model = LlavaNextVideoForConditionalGeneration.from_pretrained(
            "llava-hf/LLaVA-NeXT-Video-7B-hf",
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            device_map="auto",
        )
        processor = LlavaNextVideoProcessor.from_pretrained(
            "llava-hf/LLaVA-NeXT-Video-7B-hf"
        )
        return {"model": model, "processor": processor, "type": "llava_next_video"}

    def _load_llava_onevision(self):
        """加载 LLaVA-OneVision（轻量级）"""
        from transformers import LlavaOnevisionForConditionalGeneration, AutoProcessor

        model = LlavaOnevisionForConditionalGeneration.from_pretrained(
            "llava-hf/llava-onevision-qwen2-0.5b-ov-hf",
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            device_map="auto",
        )
        processor = AutoProcessor.from_pretrained(
            "llava-hf/llava-onevision-qwen2-0.5b-ov-hf"
        )
        return {"model": model, "processor": processor, "type": "llava_onevision"}

    def unload_model(self, model_name: str):
        """卸载模型释放显存"""
        if model_name in self.loaded_models:
            del self.loaded_models[model_name]
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print(f"♻️ 卸载模型: {model_name}")

    def get_loaded_models(self):
        """获取已加载的模型列表"""
        return list(self.loaded_models.keys())
