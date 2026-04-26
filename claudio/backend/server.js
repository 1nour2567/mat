const express = require('express');
const http = require('http');
const socketIO = require('socket.io');
const fs = require('fs-extra');
const path = require('path');
require('dotenv').config();

const app = express();
const server = http.createServer(app);
const io = socketIO(server);

// 静态文件服务
app.use(express.static(path.join(__dirname, '../frontend')));

// 解析 JSON 请求体
app.use(express.json());

// 健康检查
app.get('/api/health', (req, res) => {
  res.json({ status: 'ok', timestamp: new Date().toISOString() });
});

// 音乐推荐接口
app.get('/api/recommend', (req, res) => {
  const { time, mood, genre } = req.query;
  // 这里将连接到 Claude API 获取推荐
  res.json({
    status: 'ok',
    recommendations: [
      { title: '颜色', artist: '许美静', url: 'https://music.163.com/song?id=287248' },
      { title: '安静', artist: '周杰伦', url: 'https://music.163.com/song?id=186016' }
    ]
  });
});

// 聊天接口
app.post('/api/chat', (req, res) => {
  const { message } = req.body;
  // 这里将连接到 Claude API 处理聊天
  res.json({
    status: 'ok',
    response: '我是 Claudio，你的音乐助手。今天想听听什么类型的音乐？'
  });
});

// Socket.io 连接处理
io.on('connection', (socket) => {
  console.log('New client connected');
  
  socket.on('chat message', (msg) => {
    console.log('Message:', msg);
    // 处理聊天消息
    socket.emit('chat response', {
      message: '收到你的消息：' + msg,
      timestamp: new Date().toISOString()
    });
  });
  
  socket.on('disconnect', () => {
    console.log('Client disconnected');
  });
});

// 启动服务器
const PORT = process.env.PORT || 3000;
server.listen(PORT, () => {
  console.log(`Server running on port ${PORT}`);
});