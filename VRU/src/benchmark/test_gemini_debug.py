#!/usr/bin/env python3
"""
Gemini API 调试脚本 - 直接测试Gemini推理
"""
import sys
import os

sys.path.insert(0, '/home/24068286g/UString/VRU/src/benchmark')

print("=" * 60)
print("Gemini API 调试测试")
print("=" * 60)

# 1. 检查依赖
print("\n[1] 检查依赖...")
try:
    import google.generativeai as genai
    print("  ✓ google-generativeai 已安装")
except ImportError:
    print("  ❌ google-generativeai 未安装，正在安装...")
    os.system("pip install -q google-generativeai")
    import google.generativeai as genai
    print("  ✓ google-generativeai 已安装")

# 2. 测试API Key
print("\n[2] 测试API Key...")
API_KEY = 'AIzaSyBMGl_HlqsL433V5dDFur8_ljZud6_KNzE'
print(f"  API Key: {API_KEY[:20]}...{API_KEY[-10:]}")

try:
    genai.configure(api_key=API_KEY)
    print("  ✓ API Key 配置成功")
except Exception as e:
    print(f"  ❌ API Key 配置失败: {e}")
    sys.exit(1)

# 3. 列出可用模型
print("\n[3] 列出可用模型...")
try:
    models = genai.list_models()
    print("  可用模型：")
    for model in models:
        if 'gemini' in model.name.lower():
            print(f"    - {model.name}")
except Exception as e:
    print(f"  ❌ 列出模型失败: {e}")

# 4. 测试简单推理
print("\n[4] 测试简单推理...")
try:
    model = genai.GenerativeModel("models/gemini-2.5-pro")
    print("  ✓ 模型初始化成功")
    
    test_prompt = """请回答以下选择题：

题1: Python 是什么?
A) 编程语言
B) 蛇
C) 电影

请按照以下格式回答：
题1:A
"""
    
    print("  发送请求...")
    response = model.generate_content(
        test_prompt,
        generation_config=genai.types.GenerationConfig(
            temperature=0.1,
            max_output_tokens=128,
        )
    )
    
    text = response.text
    print(f"  ✓ 推理成功")
    print(f"  回复内容：{text[:100]}...")
    
except Exception as e:
    print(f"  ❌ 推理失败: {e}")
    import traceback
    traceback.print_exc()

# 5. 测试GeminiRunner
print("\n[5] 测试 GeminiRunner 类...")
try:
    from models.gemini_runner import GeminiRunner
    
    runner = GeminiRunner('gemini-2.5-pro')
    print("  ✓ GeminiRunner 初始化成功")
    
    prompt = """请回答以下多选题（每题选一个选项）:

Q1: 1+1等于?
选项:
1) 1
2) 2
3) 3
4) 4

Q2: 2+2等于?
选项:
1) 2
2) 3
3) 4
4) 5

Q3: 3+3等于?
选项:
1) 3
2) 4
3) 6
4) 9

Q4: 1+2等于?
选项:
1) 2
2) 3
3) 4
4) 5

Q5: 2+3等于?
选项:
1) 3
2) 4
3) 5
4) 6

Q6: 4+1等于?
选项:
1) 3
2) 4
3) 5
4) 6

请按 Q1:# Q2:# Q3:# Q4:# Q5:# Q6:# 的格式回答，其中#是数字1-4"""
    
    print("  发送推理请求...")
    result = runner.predict("test_video_001", prompt, [])
    print(f"  ✓ 推理成功")
    print(f"  结果: {result}")
    
    # 验证结果格式
    if isinstance(result, list) and len(result) == 6:
        if all(1 <= x <= 4 for x in result):
            print("  ✓ 结果格式正确")
        else:
            print(f"  ⚠️ 结果值范围可能有问题: {result}")
    else:
        print(f"  ⚠️ 结果格式错误，应该是6个数字，得到: {result}")
    
except Exception as e:
    print(f"  ❌ GeminiRunner 测试失败: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)
print("测试完成")
print("=" * 60)
