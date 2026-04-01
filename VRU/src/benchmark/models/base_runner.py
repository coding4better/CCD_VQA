import re


class BaseRunner:
    """基础 Runner（其他模型的默认实现）"""
    
    def __init__(self, model_name):
        self.model_name = model_name
        print(f"  ⚠️ {model_name} 使用默认 BaseRunner（功能有限）")
    
    def predict(self, video_number, prompt, video_frames):
        """默认推理（返回mock答案）"""
        print(f"  ⚠️ {self.model_name} 未实现具体推理逻辑，返回默认答案")
        return [1, 1, 1, 1, 1, 1]
    
    def _parse_choices(self, response: str) -> list:
        """解析选项序号"""
        choices = []
        
        pattern1 = r'(\d+):(\d)'
        matches = re.findall(pattern1, response)
        
        if matches and len(matches) >= 6:
            matches.sort(key=lambda x: int(x[0]))
            choices = [int(m[1]) for m in matches[:6]]
        else:
            numbers = re.findall(r'\d', response)
            if len(numbers) >= 6:
                choices = [int(n) for n in numbers[:6]]
        
        while len(choices) < 6:
            choices.append(1)
        
        return choices[:6]
    
    def release(self):
        """释放资源"""
        print(f"  ✓ {self.model_name} 已释放")
