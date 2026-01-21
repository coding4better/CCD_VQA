#!/usr/bin/env python3
"""
验证脚本：检查一致性检查实验的环境和数据

运行此脚本以验证：
1. 依赖包是否安装
2. 数据文件是否存在
3. API 密钥是否有效
4. 文件权限是否正确
"""

import sys
import os
from pathlib import Path
import json

def print_header(text):
    """打印标题"""
    print("\n" + "="*80)
    print(f"  {text}")
    print("="*80)

def check_python_version():
    """检查 Python 版本"""
    print_header("1. Python 版本检查")
    version = sys.version_info
    print(f"Python 版本: {version.major}.{version.minor}.{version.micro}")
    if version.major >= 3 and version.minor >= 7:
        print("✅ Python 版本满足要求 (3.7+)")
        return True
    else:
        print("❌ Python 版本过低，建议升级到 3.7+")
        return False

def check_dependencies():
    """检查依赖包"""
    print_header("2. 依赖包检查")
    
    packages = {
        'google.generativeai': 'google-generativeai',
        'pandas': 'pandas',
        'numpy': 'numpy',
        'matplotlib': 'matplotlib',
        'tqdm': 'tqdm',
    }
    
    all_ok = True
    for module, package_name in packages.items():
        try:
            __import__(module)
            print(f"✅ {package_name} 已安装")
        except ImportError:
            print(f"❌ {package_name} 未安装")
            print(f"   安装命令: pip install {package_name}")
            all_ok = False
    
    return all_ok

def check_data_files():
    """检查数据文件"""
    print_header("3. 数据文件检查")
    
    files = {
        'Baseline 描述': "/home/24068286g/CCD_VQA/VRU/src/description_generation/gemini_descriptions_20260119_062930.json",
        'QA 数据': "/home/24068286g/CCD_VQA/VRU/src/description_generation/generated_vqa_eng.json",
    }
    
    all_ok = True
    for name, path in files.items():
        file_path = Path(path)
        if file_path.exists():
            size_mb = file_path.stat().st_size / (1024 * 1024)
            print(f"✅ {name} 存在 ({size_mb:.2f} MB)")
            
            # 尝试加载并验证
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        print(f"   └─ 包含 {len(data)} 条记录")
            except Exception as e:
                print(f"❌ 文件格式错误: {e}")
                all_ok = False
        else:
            print(f"❌ {name} 不存在: {path}")
            all_ok = False
    
    return all_ok

def check_api_key():
    """检查 API 密钥"""
    print_header("4. API 密钥检查")
    
    api_key = os.environ.get('GEMINI_API_KEY')
    
    if not api_key:
        print("❌ 未设置 GEMINI_API_KEY 环境变量")
        print("   设置方法: export GEMINI_API_KEY='your_key'")
        return False
    
    if api_key == 'your_gemini_api_key_here':
        print("❌ GEMINI_API_KEY 还是默认值，需要替换")
        return False
    
    if len(api_key) < 20:
        print("❌ API 密钥看起来太短")
        return False
    
    print(f"✅ API 密钥已设置")
    print(f"   密钥长度: {len(api_key)} 字符")
    
    # 尝试验证
    try:
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        models = list(genai.list_models())
        print(f"✅ API 密钥有效 ({len(models)} 个可用模型)")
        return True
    except Exception as e:
        print(f"❌ API 密钥验证失败: {e}")
        return False

def check_output_directory():
    """检查输出目录"""
    print_header("5. 输出目录检查")
    
    output_dir = Path("/home/24068286g/CCD_VQA/VRU/src/description_check/results")
    
    if output_dir.exists():
        print(f"✅ 输出目录存在: {output_dir}")
        if os.access(output_dir, os.W_OK):
            print(f"✅ 目录可写")
            return True
        else:
            print(f"❌ 目录不可写")
            return False
    else:
        print(f"❌ 输出目录不存在: {output_dir}")
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
            print(f"✅ 已创建目录")
            return True
        except Exception as e:
            print(f"❌ 创建目录失败: {e}")
            return False

def check_script_files():
    """检查脚本文件"""
    print_header("6. 脚本文件检查")
    
    script_dir = Path("/home/24068286g/CCD_VQA/VRU/src/description_check")
    
    required_files = {
        'Python 脚本': "exp2_consistency_check.py",
        'Jupyter Notebook': "exp2_consistency_check.ipynb",
        'README': "README.md",
        'Quick Start': "QUICKSTART.md",
        'Implementation': "IMPLEMENTATION.md",
        'Usage Examples': "usage_examples.py",
    }
    
    all_ok = True
    for name, filename in required_files.items():
        file_path = script_dir / filename
        if file_path.exists():
            size_kb = file_path.stat().st_size / 1024
            print(f"✅ {name} 存在 ({size_kb:.1f} KB)")
        else:
            print(f"❌ {name} 不存在: {filename}")
            all_ok = False
    
    return all_ok

def check_permissions():
    """检查文件权限"""
    print_header("7. 文件权限检查")
    
    script_file = Path("/home/24068286g/CCD_VQA/VRU/src/description_check/exp2_consistency_check.py")
    
    if script_file.exists():
        is_readable = os.access(script_file, os.R_OK)
        is_writable = os.access(script_file, os.W_OK)
        
        print(f"✅ 脚本文件存在")
        print(f"  可读: {'✅' if is_readable else '❌'}")
        print(f"  可写: {'✅' if is_writable else '❌'}")
        
        return is_readable
    else:
        print("❌ 脚本文件不存在")
        return False

def print_summary(results):
    """打印总结"""
    print_header("检查总结")
    
    checks = [
        ("Python 版本", results[0]),
        ("依赖包", results[1]),
        ("数据文件", results[2]),
        ("API 密钥", results[3]),
        ("输出目录", results[4]),
        ("脚本文件", results[5]),
        ("文件权限", results[6]),
    ]
    
    passed = sum(1 for _, result in checks if result)
    total = len(checks)
    
    print(f"\n检查结果: {passed}/{total} 通过\n")
    
    for name, result in checks:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {status:8} {name}")
    
    print()
    
    if passed == total:
        print("🎉 所有检查都通过！可以开始使用实验脚本。")
        return True
    else:
        print(f"⚠️  还有 {total - passed} 个检查未通过。")
        print("   请根据上面的提示进行修复。")
        return False

def print_next_steps():
    """打印后续步骤"""
    print_header("后续步骤")
    
    print("""
1. 快速启动 (Google Colab, 推荐):
   - 打开 exp2_consistency_check.ipynb
   - 替换 API 密钥
   - 逐个运行单元格

2. 本地运行 (Python):
   cd /home/24068286g/CCD_VQA/VRU/src/description_check
   python exp2_consistency_check.py

3. 查看结果:
   ls -la results/
   # 查看生成的文件

4. 详细信息:
   - 快速指南: QUICKSTART.md
   - 完整文档: README.md
   - 实现细节: IMPLEMENTATION.md
   - 代码示例: usage_examples.py

5. 修改参数 (如需要):
   - 采样大小: sample_size = 10
   - 延迟时间: time.sleep(0.5)
   - 模型选择: model_name = "gemini-2.0-flash"

6. 获取帮助:
   - 查看日志输出
   - 参考 README 中的常见问题
   - 检查 API 错误信息

祝您实验顺利！🚀
    """)

def main():
    """主函数"""
    print("\n" + "▄"*80)
    print("█  一致性检查实验 - 环境验证脚本")
    print("█  Exp2: Description Consistency Check")
    print("▀"*80)
    
    # 执行所有检查
    results = [
        check_python_version(),
        check_dependencies(),
        check_data_files(),
        check_api_key(),
        check_output_directory(),
        check_script_files(),
        check_permissions(),
    ]
    
    # 打印总结
    all_pass = print_summary(results)
    
    # 打印后续步骤
    print_next_steps()
    
    return 0 if all_pass else 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
