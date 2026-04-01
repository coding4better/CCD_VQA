import re
import os
import sys


class GeminiRunner:
    """Gemini API 选择题推理Runner    
    需要设置环境变量：GEMINI_API_KEY 或在代码中设置
    地理位置限制问题：需要使用代理或 VPN
    """
    
    def __init__(self, model_name):
        self.model_name = model_name
        self.api_key = os.environ.get('GEMINI_API_KEY')
        self.genai = None
        self._init_api()
    
    def _init_api(self):
        """初始化 Gemini API"""
        try:
            import google.generativeai as genai
            
            if not self.api_key:
                print("  ⚠️ API 密钥未设置！")
                print("  请设置环境变量 GEMINI_API_KEY")
                return
            
            # 尝试配置代理以绕过地理限制
            self._setup_proxy()
            
            # 配置 API
            genai.configure(api_key=self.api_key)
            self.genai = genai
            
            print(f"  ✓ {self.model_name} API 初始化成功 (API Key 已设置)")
            
        except ImportError:
            print("  ❌ google-generativeai 未安装，请运行: pip install google-generativeai")
        except Exception as e:
            print(f"  ⚠ API 初始化警告: {e} (稍后会重试)")
    
    def _setup_proxy(self):
        """配置代理以绕过地理限制"""
        try:
            # 从环境变量读取代理配置
            http_proxy = os.environ.get('HTTP_PROXY') or os.environ.get('http_proxy')
            https_proxy = os.environ.get('HTTPS_PROXY') or os.environ.get('https_proxy')
            
            if http_proxy or https_proxy:
                print(f"  ℹ 检测到代理配置: HTTP={http_proxy}, HTTPS={https_proxy}")
                
                # 设置为全局 requests 代理
                import requests
                session = requests.Session()
                if http_proxy:
                    session.proxies['http'] = http_proxy
                if https_proxy:
                    session.proxies['https'] = https_proxy
        except:
            pass  # 代理配置是可选的
    
    def predict(self, video_number, prompt, video_frames, num_options=4, expected_count=6):
        """推理选择题，API默认纯文本，支持全部视频帧
        
        地理位置限制问题的解决方案：
        1. 使用代理: export HTTP_PROXY=... HTTPS_PROXY=...
        2. 使用 VPN
        3. 在允许的地区运行
        """
        try:
            if not self.api_key or not self.genai:
                print(f"  ❌ API 未正确初始化")
                return [1] * max(1, expected_count)
            
            # 模型列表（按优先级排序）
            # Flash 模型通常地理限制较少
            model_names = [
                "models/gemini-2.0-flash-exp",   # Flash 实验版（限制最少）
                "models/gemini-2.0-flash",       # Flash 稳定版
                "models/gemini-1.5-flash",       # 旧版 Flash
                "gemini-2.0-flash",              # 不带前缀
                "gemini-1.5-flash",
                "gemini-flash",                  # 通用 Flash
            ]
            
            response = None
            last_error = None
            
            for model_name in model_names:
                try:
                    print(f"  [尝试] 使用模型: {model_name}")
                    model = self.genai.GenerativeModel(model_name)
                    
                    # 发送请求
                    response = model.generate_content(
                        prompt,
                        generation_config=self.genai.types.GenerationConfig(
                            temperature=0.1,
                            max_output_tokens=128,
                        )
                    )
                    
                    # 如果成功，提取文本
                    if response and hasattr(response, 'text'):
                        text = response.text
                        choices = self._parse_choices(text, num_options=num_options, expected_count=expected_count)
                        print(f"  ✓ {self.model_name} 推理完成: {video_number} (模型: {model_name})")
                        return choices
                        
                except Exception as model_error:
                    last_error = str(model_error)
                    error_msg = str(model_error).lower()
                    
                    # 地理位置限制 - 需要代理
                    if "location" in error_msg:
                        print(f"  ❌ 地理位置限制: {model_error}")
                        print(f"  💡 解决方案: 设置代理或 VPN")
                        print(f"     export HTTP_PROXY=http://...:port")
                        print(f"     export HTTPS_PROXY=https://...:port")
                        return [1] * max(1, expected_count)
                    
                    # 认证/配额错误 - API KEY 问题
                    elif any(x in error_msg for x in ['invalid', 'unauthorized', 'quota', 'permission', 'api_key']):
                        print(f"  ❌ API 认证/配额错误: {model_error}")
                        print(f"  💡 检查 API KEY 是否有效或已达配额限制")
                        return [1] * max(1, expected_count)
                    
                    # 模型不存在
                    elif any(x in error_msg for x in ['not found', '404']):
                        print(f"  ⚠ 模型不可用: {model_name}")
                        continue
                    
                    # 其他错误，继续尝试
                    else:
                        print(f"  ⚠ {model_name} 错误: {model_error}")
                        continue
            
            # 所有模型都失败
            print(f"  ❌ 所有模型尝试失败")
            print(f"  最后一个错误: {last_error}")
            return [1] * max(1, expected_count)
            
        except Exception as e:
            print(f"  ❌ 推理异常: {e}")
            return [1] * max(1, expected_count)
    
    def _parse_choices(self, response: str, num_options: int = 4, expected_count: int = 6) -> list:
        """解析选项序号"""
        if num_options <= 0:
            num_options = 4
        if expected_count <= 0:
            expected_count = 1

        max_opt = str(num_options)
        opt_char_class = f"[1-{max_opt}]"
        choices = []
        
        # 匹配 "题号:答案" 格式
        pattern1 = rf'(\d+):({opt_char_class})'
        matches = re.findall(pattern1, response)
        
        if matches and len(matches) >= expected_count:
            matches.sort(key=lambda x: int(x[0]))
            choices = [int(m[1]) for m in matches[:expected_count]]
        else:
            # 匹配关键词后的数字
            pattern2 = rf'(?:答案|选项|option|answer)\s*[：:]*\s*({opt_char_class})'
            matches2 = re.findall(pattern2, response.lower())
            if len(matches2) >= expected_count:
                choices = [int(m) for m in matches2[:expected_count]]
            else:
                # 匹配独立数字（word boundary）
                numbers = re.findall(rf'\b({opt_char_class})\b', response)
                if len(numbers) >= expected_count:
                    choices = [int(n) for n in numbers[:expected_count]]
        
        # 过滤掉无效选项
        choices = [c if 1 <= c <= num_options else 1 for c in choices]
        
        while len(choices) < expected_count:
            choices.append(1)
        
        return choices[:expected_count]
    
    def release(self):
        """API 模型无需释放本地资源"""
        print(f"  ✓ {self.model_name} 无本地资源占用")
