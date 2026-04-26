# Claudio - AI 音乐代理

Claudio 是一个智能音乐代理系统，能够根据用户的历史歌单和实时场景推荐音乐，并通过聊天界面与用户互动。

## 核心功能

- **个性化推荐**：根据用户历史歌单和实时场景（如周一晚上）推荐音乐
- **多维度联动**：前端为播放器 + 聊天界面，后端连接音响设备、音乐 API 和 Claude Code
- **动态调整**：用户可随时通过聊天界面与 Claudio 互动，调整歌单或获取音乐推荐

## 技术架构

- **前端**：HTML5 + CSS3 + JavaScript，支持响应式设计
- **后端**：Node.js + Express + Socket.io
- **API 集成**：Claude API、音乐 API、天气 API 等

## 项目结构

```
claudio/
├── frontend/          # 前端代码
│   └── index.html     # 主页面
├── backend/           # 后端代码
│   ├── server.js      # 主服务器
│   ├── package.json   # 依赖配置
│   └── .env           # 环境变量
└── README.md          # 项目说明
```

## 快速开始

### 1. 安装依赖

```bash
cd backend
npm install
```

### 2. 配置环境变量

编辑 `backend/.env` 文件，添加必要的 API 密钥：

```env
# Claude API 配置
CLAUDE_API_KEY=your_claude_api_key

# 音乐 API 配置
MUSIC_API_KEY=your_music_api_key

# 其他 API 配置
WEATHER_API_KEY=your_weather_api_key
CALENDAR_API_KEY=your_calendar_api_key
```

### 3. 启动服务器

```bash
npm start
```

或使用开发模式：

```bash
npm run dev
```

### 4. 访问应用

打开浏览器，访问 `http://localhost:3000`

## 功能说明

### 播放器界面
- 显示当前播放的歌曲信息
- 提供播放/暂停、上一曲、下一曲控制
- 显示播放进度

### 聊天界面
- 与 Claudio 进行自然语言对话
- 询问音乐推荐
- 调整播放列表
- 了解当前音乐信息

## API 集成

- **Claude API**：处理自然语言对话和音乐推荐逻辑
- **音乐 API**：获取歌曲信息和播放链接
- **天气 API**：根据天气情况调整音乐推荐
- **日历 API**：根据日程安排调整音乐推荐

## 未来计划

- 支持更多音乐平台
- 添加语音控制功能
- 实现个性化音乐学习算法
- 支持多设备同步

## 注意事项

- 本项目需要有效的 API 密钥才能正常运行
- 部分功能可能需要额外的硬件支持（如音响设备）
- 请确保网络连接稳定以获得最佳体验