from peft import LoraConfig
from transformers import Blip2ForConditionalGeneration

# 加载最小化模型（提案中的BLIP-2）
model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b", device_map="auto")

# 配置LoRA（按提案参数）
lora_config = LoraConfig(
    r=8,  # 提案建议的秩
    target_modules=["q_proj", "v_proj"],  # 仅微调Q-Former的特定层
    lora_alpha=32,
    lora_dropout=0.05
)
model.add_adapter(lora_config)

# 测试显存占用
import torch
dummy_input = torch.randn(1, 3, 224, 224).to("cuda")
with torch.no_grad():
    print(torch.cuda.max_memory_allocated() / 1024**3)  # 应<18GB