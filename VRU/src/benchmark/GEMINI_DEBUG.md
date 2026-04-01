# Gemini API Benchmark 调试指南

## 当前状态

✓ **代码已修复并准备就绪**
✓ **所有依赖包已安装**  
✗ **地理位置限制阻止 API 调用**

---

## 问题分析

### 错误信息
```
400 User location is not supported for the API use.
```

### 原因
Google Gemini API 对某些地理位置有限制。由于您的服务器位置不在支持的地区，无法直接调用 API。

---

## 解决方案

### 方案 1: 使用代理服务器 (推荐)

#### A. 使用免费公共代理

1. **获取代理列表**
   - 访问: https://free-proxy-list.net/
   - 选择来自"支持的地区"的代理 (如美国、欧洲)
   - 复制 IP 地址和端口号

2. **设置环境变量**
   ```bash
   export HTTP_PROXY=http://代理IP:端口
   export HTTPS_PROXY=https://代理IP:端口
   ```

3. **验证代理设置**
   ```bash
   cd /home/24068286g/UString/VRU/src/benchmark
   python models/gemini_debug.py
   ```

4. **运行 Benchmark**
   ```bash
   python run_benchmark.py
   ```

#### B. 使用付费代理服务

推荐的付费代理服务:
- **Bright Data** (https://brightdata.com/)
- **Smartproxy** (https://smartproxy.com/)
- **Oxylabs** (https://oxylabs.io/)
- **Proxy-provider** (https://proxy-provider.com/)

设置方式:
```bash
export HTTP_PROXY=http://username:password@代理IP:端口
export HTTPS_PROXY=https://username:password@代理IP:端口
```

---

### 方案 2: 使用 VPN

1. **安装 VPN 客户端**
   - OpenVPN
   - WireGuard
   - 商业 VPN (ExpressVPN, NordVPN, ProtonVPN 等)

2. **连接到支持的地区**
   - 选择美国、欧洲或其他支持的地区
   - 建立 VPN 连接

3. **验证连接**
   ```bash
   python models/gemini_debug.py
   ```

4. **运行 Benchmark**
   ```bash
   python run_benchmark.py
   ```

---

### 方案 3: 替代 API 提供商

如果以上方案都不可行，考虑使用其他 API：

1. **OpenAI (GPT)**
   - 支持的地区更广
   - API: https://api.openai.com/

2. **Azure OpenAI**
   - 企业级支持
   - 更好的地理覆盖

3. **Anthropic (Claude)**
   - API: https://www.anthropic.com/

---

## 快速测试

### 检查当前代理设置
```bash
echo $HTTP_PROXY
echo $HTTPS_PROXY
```

### 测试代理连接
```bash
curl -x $HTTP_PROXY https://www.google.com -I
```

### 运行诊断工具
```bash
cd /home/24068286g/UString/VRU/src/benchmark
python models/gemini_debug.py
```

### 如果代理设置成功，应该看到
```
✓ 代理设置
✓ Gemini API
✓ GeminiRunner
```

---

## 代码改进总结

已对 `gemini_runner.py` 进行了以下改进:

### 1. **更好的错误处理**
```python
- 地理位置限制检测
- API KEY 有效性检查
- 模型可用性检查
```

### 2. **灵活的模型支持**
```python
- 自动尝试多个模型版本
- 备用模型列表
- 优雅降级处理
```

### 3. **诊断工具**
```python
- gemini_debug.py: 完整诊断脚本
- gemini_proxy_setup.py: 代理配置指南
```

### 4. **环境变量支持**
```python
- 支持 GEMINI_API_KEY 环境变量
- 支持 HTTP_PROXY 和 HTTPS_PROXY
- 支持 API KEY 硬编码作为备用
```

---

## 下一步

### 最推荐的步骤:

1. **选择代理服务**
   ```bash
   # 尝试免费代理 (https://free-proxy-list.net/)
   export HTTP_PROXY=http://1.1.1.1:8080
   export HTTPS_PROXY=https://1.1.1.1:8080
   ```

2. **验证代理**
   ```bash
   python models/gemini_debug.py
   ```

3. **如果诊断成功，运行 Benchmark**
   ```bash
   python run_benchmark.py
   ```

---

## 文件列表

- `gemini_runner.py` - 主要的 Gemini API 运行器类
- `gemini_debug.py` - 诊断和故障排查工具
- `gemini_proxy_setup.py` - 代理配置指南
- `GEMINI_DEBUG.md` - 本文件

---

## 常见问题

### Q1: 如何验证 API KEY 是否有效?
```bash
python models/gemini_debug.py
```
如果看到 "✓ API Key 已设置" 且 "✓ API 初始化成功"，说明 API KEY 有效。

### Q2: 代理设置后仍然不工作?
- 检查代理 IP 和端口是否正确
- 尝试不同的代理服务器
- 确保代理在支持的地区（美国、欧洲等）

### Q3: 如何找到好的免费代理?
- https://free-proxy-list.net/
- https://www.proxy-list.download/
- 选择高速、长时间在线的代理

### Q4: 是否可以不使用代理?
- 如果您的服务器在支持的地区（如北美、欧洲），可以直接使用
- 否则必须使用代理或 VPN

---

## 支持

如有其他问题，请查看:
- Gemini API 文档: https://ai.google.dev/
- 代理服务文档: 对应服务商的官方文档

