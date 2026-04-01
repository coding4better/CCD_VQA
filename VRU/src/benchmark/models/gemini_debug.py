#!/usr/bin/env python3
"""
Gemini API Benchmark 调试脚本
完整的诊断和问题排查工具
"""

import sys
import os
sys.path.insert(0, '/home/24068286g/UString/VRU/src/benchmark')


def print_section(title):
    """打印分隔线"""
    print("\n" + "=" * 75)
    print(f"  {title}")
    print("=" * 75)


def check_dependencies():
    """检查依赖包"""
    print_section("1. 检查依赖包")
    
    required_packages = {
        'google.generativeai': 'google-generativeai',
        're': 're',
    }
    
    all_ok = True
    for module_name, package_name in required_packages.items():
        try:
            __import__(module_name)
            print(f"  ✓ {package_name:30} - 已安装")
        except ImportError:
            print(f"  ✗ {package_name:30} - 缺失")
            all_ok = False
    
    return all_ok


def check_api_key():
    """检查 API KEY"""
    print_section("2. 检查 API KEY")
    
    api_key = os.environ.get('GEMINI_API_KEY') or 'AIzaSyBMGl_HlqsL433V5dDFur8_ljZud6_KNzE'
    
    if api_key:
        # 隐藏大部分 key
        masked = f"{api_key[:8]}...{api_key[-8:]}"
        print(f"  ✓ API KEY 已设置: {masked}")
        return True
    else:
        print(f"  ✗ API KEY 未设置")
        print(f"    设置方法: export GEMINI_API_KEY=your_api_key")
        return False


def check_network():
    """检查网络连接"""
    print_section("3. 检查网络连接")
    
    try:
        import socket
        socket.create_connection(("8.8.8.8", 53), timeout=3)
        print(f"  ✓ 基础网络连接正常")
        return True
    except Exception as e:
        print(f"  ✗ 网络连接失败: {e}")
        return False


def check_proxy():
    """检查代理设置"""
    print_section("4. 检查代理设置")
    
    http_proxy = os.environ.get('HTTP_PROXY') or os.environ.get('http_proxy')
    https_proxy = os.environ.get('HTTPS_PROXY') or os.environ.get('https_proxy')
    
    if http_proxy:
        print(f"  ℹ HTTP_PROXY: {http_proxy}")
    else:
        print(f"  ℹ HTTP_PROXY: 未设置")
    
    if https_proxy:
        print(f"  ℹ HTTPS_PROXY: {https_proxy}")
    else:
        print(f"  ℹ HTTPS_PROXY: 未设置")
    
    if http_proxy or https_proxy:
        print(f"\n  ✓ 已配置代理")
        return True
    else:
        print(f"\n  ⚠ 未配置代理（可能需要用于绕过地理限制）")
        return False


def test_gemini_api():
    """测试 Gemini API"""
    print_section("5. 测试 Gemini API")
    
    try:
        import google.generativeai as genai
        
        api_key = os.environ.get('GEMINI_API_KEY') or 'AIzaSyBMGl_HlqsL433V5dDFur8_ljZud6_KNzE'
        genai.configure(api_key=api_key)
        print(f"  ✓ API 初始化成功")
        
        # 尝试列出模型
        print(f"  尝试列出可用模型...")
        models = list(genai.list_models())
        print(f"  ✓ 成功获取 {len(models)} 个模型")
        for m in models[:3]:
            print(f"    - {m.name}")
        return True
        
    except Exception as e:
        error_msg = str(e)
        print(f"  ✗ API 测试失败: {error_msg}")
        
        if "location" in error_msg.lower():
            print(f"\n  🔴 问题: 地理位置限制")
            print(f"\n  解决方案:")
            print(f"     1. 使用代理或 VPN")
            print(f"     2. 设置环境变量: export HTTP_PROXY=http://...:port")
            print(f"     3. 或设置 HTTPS_PROXY: export HTTPS_PROXY=https://...:port")
        elif "invalid" in error_msg.lower() or "unauthorized" in error_msg.lower():
            print(f"\n  🔴 问题: API KEY 无效或未授权")
            print(f"\n  解决方案:")
            print(f"     1. 检查 API KEY 是否正确")
            print(f"     2. 确保 API KEY 有 Gemini API 权限")
            print(f"     3. 检查配额是否已用完")
        
        return False


def test_gemini_runner():
    """测试 GeminiRunner 类"""
    print_section("6. 测试 GeminiRunner 类")
    
    try:
        from models.gemini_runner import GeminiRunner
        
        runner = GeminiRunner('gemini-2.5-pro')
        print(f"  ✓ GeminiRunner 初始化成功")
        
        # 简单测试
        test_prompt = "你好，请回答：1+1等于几？"
        result = runner.predict("test", test_prompt, [])
        
        print(f"  ✓ predict 方法执行成功")
        print(f"    结果: {result}")
        
        return result != [1, 1, 1, 1, 1, 1]
        
    except Exception as e:
        print(f"  ✗ GeminiRunner 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def print_summary():
    """打印诊断总结"""
    print_section("诊断总结")
    
    results = {
        "依赖包": check_dependencies(),
        "API KEY": check_api_key(),
        "网络连接": check_network(),
        "代理设置": check_proxy(),
    }
    
    api_ok = test_gemini_api()
    results["Gemini API"] = api_ok
    
    if api_ok:
        results["GeminiRunner"] = test_gemini_runner()
    else:
        print("\n  ⏭️ 跳过 GeminiRunner 测试（Gemini API 不可用）")
    
    print_section("结论")
    
    failed = [k for k, v in results.items() if not v]
    
    if not failed:
        print("\n  ✓ 所有检查通过！")
        print("\n  现在可以运行 benchmark:")
        print("  cd /home/24068286g/UString/VRU/src/benchmark")
        print("  python run_benchmark.py")
    else:
        print(f"\n  ✗ 发现 {len(failed)} 个问题:")
        for issue in failed:
            print(f"    - {issue}")
        
        print(f"\n  主要问题分析:")
        if "Gemini API" in failed:
            print(f"    • API 连接失败 - 需要代理/VPN 或检查 API KEY")
        
        print(f"\n  建议的解决步骤:")
        print(f"    1. 确保网络连接正常")
        print(f"    2. 安装所有依赖包: pip install google-generativeai")
        print(f"    3. 检查或获取有效的 API KEY")
        print(f"    4. 如果遇到地理限制，设置代理:")
        print(f"       export HTTP_PROXY=http://PROXY_IP:PORT")
        print(f"       export HTTPS_PROXY=https://PROXY_IP:PORT")
        print(f"    5. 重新运行此诊断脚本")


def main():
    """主函数"""
    print("\n")
    print("╔" + "=" * 73 + "╗")
    print("║" + " " * 73 + "║")
    print("║" + "  Gemini API Benchmark 诊断工具".center(73) + "║")
    print("║" + " " * 73 + "║")
    print("╚" + "=" * 73 + "╝")
    
    print_summary()
    
    print("\n" + "=" * 75)
    print("诊断完成")
    print("=" * 75 + "\n")


if __name__ == '__main__':
    main()
