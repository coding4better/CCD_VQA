"""
Gemini API 代理配置指南

问题: 400 User location is not supported for the API use.
原因: Google Gemini API 对某些地区有地理位置限制

解决方案:
"""

import os
import subprocess
import sys


def setup_proxy_for_gemini():
    """配置代理环境变量"""
    
    print("=" * 70)
    print("Gemini API 地理位置限制解决方案")
    print("=" * 70)
    
    print("\n方案 1: 使用免费公共代理 (推荐先尝试)")
    print("-" * 70)
    print("""
# 获取可用的代理列表（来自 https://free-proxy-list.net/）
# 选择一个地址不受限制的代理IP和端口

# 设置代理环境变量:
export HTTP_PROXY=http://34.216.224.9:40715
export HTTPS_PROXY=https://34.216.224.9:40715

# 然后运行 benchmark:
cd /home/24068286g/UString/VRU/src/benchmark
python run_benchmark.py
    """)
    
    print("\n方案 2: 使用 VPN")
    print("-" * 70)
    print("""
# 使用 OpenVPN, WireGuard 或其他 VPN 工具连接到允许的地区
# VPN 连接后，会自动替换您的 IP 地址

# 常用 VPN:
# - ExpressVPN
# - NordVPN
# - ProtonVPN
# - 其他商业 VPN
    """)
    
    print("\n方案 3: 使用专业代理服务")
    print("-" * 70)
    print("""
# 使用付费代理服务，获得更稳定的连接:
# - Bright Data
# - Smartproxy  
# - Oxylabs
# 等

# 设置代理:
export HTTP_PROXY=http://username:password@PROXY_IP:PORT
export HTTPS_PROXY=https://username:password@PROXY_IP:PORT
    """)
    
    print("\n方案 4: 使用 API 代理中介")
    print("-" * 70)
    print("""
# 一些服务提供 Gemini API 的代理访问，帮助绕过地理限制:
# - 例如: https://api.pawan.krd/ (需要检查当前可用性)

# 或者在 Python 中动态切换代理:
    """)
    
    print("\n测试当前代理设置")
    print("-" * 70)
    print(f"HTTP_PROXY: {os.environ.get('HTTP_PROXY', '未设置')}")
    print(f"HTTPS_PROXY: {os.environ.get('HTTPS_PROXY', '未设置')}")
    
    
def test_proxy_connectivity():
    """测试代理连接"""
    print("\n测试代理连接...")
    
    http_proxy = os.environ.get('HTTP_PROXY')
    https_proxy = os.environ.get('HTTPS_PROXY')
    
    if not http_proxy and not https_proxy:
        print("⚠ 没有设置代理环境变量")
        return False
    
    # 测试连接
    try:
        import requests
        response = requests.get('https://www.google.com', timeout=5)
        if response.status_code == 200:
            print("✓ 代理连接正常")
            return True
    except Exception as e:
        print(f"✗ 代理连接失败: {e}")
        return False
    
    return False


def apply_proxy_to_session():
    """在 requests session 中应用代理"""
    import requests
    
    session = requests.Session()
    
    http_proxy = os.environ.get('HTTP_PROXY')
    https_proxy = os.environ.get('HTTPS_PROXY')
    
    if http_proxy:
        session.proxies['http'] = http_proxy
    if https_proxy:
        session.proxies['https'] = https_proxy
    
    return session


if __name__ == '__main__':
    # 注意: 在 Python 中设置环境变量需要使用 os.environ
    # 在 shell 中使用: export HTTP_PROXY=http://34.216.224.9:40715
    setup_proxy_for_gemini()
    test_proxy_connectivity()
