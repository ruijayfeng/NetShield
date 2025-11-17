# NetShield 部署指南

本文档详细介绍如何将NetShield部署到各种云平台，包括通过Cloudflare进行优化。

## 🌟 推荐部署方案

### 方案一：Railway + Cloudflare（推荐）

**优势**：免费额度充足、部署简单、自动HTTPS、与Cloudflare完美集成

#### 1. Railway部署步骤

1. **注册Railway账号**
   ```bash
   # 访问 https://railway.app
   # 使用GitHub账号登录
   ```

2. **创建新项目**
   - 选择"Deploy from GitHub repo"
   - 连接你的NetShield仓库
   - Railway会自动检测到`railway.json`配置

3. **配置环境变量**
   ```bash
   AI_API_KEY=your_zhipu_ai_api_key_here
   PYTHONPATH=/app/src
   ```

4. **部署**
   - Railway会自动构建和部署
   - 获得类似 `https://netshield-production.up.railway.app` 的URL

#### 2. Cloudflare优化

1. **添加自定义域名**
   - 在Cloudflare添加域名记录
   - 设置CNAME指向Railway提供的URL

2. **配置Cloudflare设置**
   ```yaml
   # cloudflare配置建议
   SSL/TLS: Full (strict)
   Security Level: Medium
   Browser Cache TTL: 4 hours
   Always Use HTTPS: On
   ```

### 方案二：Render + Cloudflare

#### 1. Render部署

1. **连接GitHub仓库**
   - 访问 https://render.com
   - 创建新的Web Service
   - 连接GitHub仓库

2. **配置构建设置**
   ```yaml
   Build Command: pip install -r requirements.txt
   Start Command: streamlit run src/visualization/dashboard.py --server.port=$PORT --server.address=0.0.0.0 --server.headless=true
   ```

3. **设置环境变量**
   ```bash
   AI_API_KEY=your_zhipu_ai_api_key
   PYTHON_VERSION=3.11.0
   ```

### 方案三：Docker + 任意云平台

#### 1. 本地测试

```bash
# 构建Docker镜像
docker build -t netshield .

# 运行容器
docker run -p 8501:8501 -e AI_API_KEY=your_api_key netshield
```

#### 2. 部署到云平台

**Google Cloud Run**：
```bash
# 构建并推送镜像
gcloud builds submit --tag gcr.io/PROJECT_ID/netshield

# 部署
gcloud run deploy netshield --image gcr.io/PROJECT_ID/netshield --platform managed --region us-central1 --allow-unauthenticated
```

**AWS ECS/Fargate**：
```bash
# 推送到ECR
aws ecr get-login-password --region us-west-2 | docker login --username AWS --password-stdin 123456789012.dkr.ecr.us-west-2.amazonaws.com

docker tag netshield:latest 123456789012.dkr.ecr.us-west-2.amazonaws.com/netshield:latest
docker push 123456789012.dkr.ecr.us-west-2.amazonaws.com/netshield:latest
```

### 方案四：Cloudflare Tunnel（本地/VPS）

如果你有自己的服务器或想要本地运行：

#### 1. 安装Cloudflare Tunnel

```bash
# 下载cloudflared
curl -L --output cloudflared.deb https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64.deb
sudo dpkg -i cloudflared.deb

# 登录
cloudflared tunnel login
```

#### 2. 创建隧道

```bash
# 创建隧道
cloudflared tunnel create netshield

# 配置DNS
cloudflared tunnel route dns netshield netshield.yourdomain.com

# 创建配置文件
cat > ~/.cloudflared/config.yml << EOF
tunnel: netshield
credentials-file: /home/user/.cloudflared/netshield.json

ingress:
  - hostname: netshield.yourdomain.com
    service: http://localhost:8501
  - service: http_status:404
EOF

# 运行隧道
cloudflared tunnel run netshield
```

#### 3. 启动NetShield

```bash
# 在另一个终端启动应用
streamlit run src/visualization/dashboard.py --server.port=8501
```

## 🔧 部署后配置

### 1. 环境变量设置

所有部署方案都需要设置以下环境变量：

```bash
AI_API_KEY=aff784bff0af47d6afa837ba011205d9.yHvAMlxIm4b2hNk1
PYTHONPATH=/app/src
PORT=8501  # 某些平台需要
```

### 2. Cloudflare优化设置

#### 安全设置
```yaml
Security Level: Medium
WAF: Custom Rules for API endpoints
Rate Limiting:
  - /api/*: 100 requests per minute
  - /*: 500 requests per minute
```

#### 性能优化
```yaml
Caching:
  - Static files: 1 year
  - HTML: 4 hours
  - API responses: No cache

Compression: Gzip + Brotli
Minify: HTML, CSS, JavaScript
```

#### 页面规则
```yaml
# API缓存规则
api.yourdomain.com/*
  - Cache Level: Bypass
  - Security Level: High

# 静态资源缓存
*.yourdomain.com/*.js
*.yourdomain.com/*.css
*.yourdomain.com/*.png
  - Cache Level: Cache Everything
  - Edge Cache TTL: 1 year
```

### 3. 监控和日志

#### Railway监控
- 使用Railway内置监控
- 设置资源使用告警

#### Render监控
```yaml
Health Check Path: /
Health Check Command: curl -f http://localhost:$PORT/ || exit 1
```

#### Cloudflare Analytics
- 启用Web Analytics
- 设置自定义事件追踪

## 🚨 故障排除

### 常见问题

#### 1. 应用启动失败
```bash
# 检查日志
# Railway: 在仪表板中查看部署日志
# Render: 在服务页面查看日志

# 常见原因：
- 缺少环境变量
- 依赖安装失败
- 端口配置错误
```

#### 2. AI功能不工作
```bash
# 检查API密钥
- 确认AI_API_KEY环境变量已设置
- 测试API连接性
- 检查智谱AI账户余额
```

#### 3. 静态文件加载问题
```bash
# Streamlit配置
# 确保.streamlit/config.toml正确配置
# 检查Cloudflare缓存设置
```

### 性能优化

#### 1. 资源限制
```yaml
Railway:
  - Memory: 512MB (免费)
  - CPU: 0.5 vCPU

Render:
  - Memory: 512MB (免费)
  - CPU: 0.5 vCPU
```

#### 2. 缓存策略
```python
# 在dashboard.py中添加缓存
@st.cache_data(ttl=300)  # 5分钟缓存
def get_system_context():
    # 原有逻辑
    pass
```

## 📊 成本估算

### 免费部署方案
- **Railway**: 免费 $5/月额度
- **Render**: 免费 750小时/月
- **Cloudflare**: 免费CDN和DNS
- **总成本**: $0/月（轻量使用）

### 生产环境方案
- **Railway Pro**: $20/月
- **Render Pro**: $25/月
- **Cloudflare Pro**: $25/月
- **总成本**: $45-70/月

## 🔒 安全建议

1. **API密钥管理**
   - 使用平台的秘密管理
   - 定期轮换API密钥
   - 监控API使用量

2. **访问控制**
   - 配置Cloudflare Access（如需要）
   - 设置IP白名单
   - 启用HTTPS严格模式

3. **数据保护**
   - 不在前端暴露敏感信息
   - 定期备份配置
   - 监控异常访问

---

选择最适合你需求的部署方案开始部署吧！如有问题，请参考各平台的官方文档或创建GitHub Issue。