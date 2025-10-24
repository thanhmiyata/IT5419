import React, { useEffect, useState } from 'react'
import { useParams } from 'react-router-dom'
import {
  Box,
  Typography,
  Paper,
  TextField,
  IconButton,
  List,
  ListItem,
  Avatar,
  Chip,
  Divider,
} from '@mui/material'
import {
  Send as SendIcon,
  SmartToy as BotIcon,
  Person as PersonIcon,
} from '@mui/icons-material'
import { ChatMessage } from '@/types'
import LoadingSpinner from '@/components/UI/LoadingSpinner'
import ChartViewport from '@/components/ChartViewport'

const ChatPage: React.FC = () => {
  const { sessionId } = useParams<{ sessionId: string }>()
  
  // Mock data for demo
  const currentSession = { id: 'demo-session', title: 'Demo Chat', messages: [] }
  const sessions = [currentSession]
  const isLoading = false
  const isTyping = false
  
  const [message, setMessage] = useState('')
  const [messages, setMessages] = useState<ChatMessage[]>([])
  const [showChart, setShowChart] = useState(false)
  const [chartSymbol, setChartSymbol] = useState('VN-INDEX')

  useEffect(() => {
    // Mock chat sessions for demo
    console.log('Loading chat sessions...')
  }, [])

  useEffect(() => {
    if (sessionId && sessions.length > 0) {
      const session = sessions.find(s => s.id === sessionId)
      if (session) {
        setMessages(session.messages)
      }
    } else if (!sessionId && sessions.length === 0) {
      // Create a new session if none exists
      console.log('Creating new chat session...')
    }
  }, [sessionId, sessions])

  const handleSendMessage = async () => {
    if (!message.trim() || !currentSession) return

    const userMessage: ChatMessage = {
      id: Date.now().toString(),
      content: message.trim(),
      type: 'text',
      sender: 'user',
      timestamp: new Date().toISOString(),
    }

    setMessages(prev => [...prev, userMessage])
    const currentMessage = message.trim().toLowerCase()
    setMessage('')

    // Check if user wants to see chart
    if (currentMessage.includes('biểu đồ') || currentMessage.includes('chart') || currentMessage.includes('đồ thị')) {
      setShowChart(true)
      
      // Extract symbol from message if possible
      const symbolMatch = currentMessage.match(/(vn-index|vni|hose|hnx|upcom)/i)
      if (symbolMatch) {
        setChartSymbol(symbolMatch[0].toUpperCase())
      }
      
      // Add bot response
      setTimeout(() => {
        const botMessage: ChatMessage = {
          id: (Date.now() + 1).toString(),
          content: `Tôi đã mở biểu đồ ${chartSymbol} cho bạn. Bạn có thể xem phân tích kỹ thuật ở bên phải.`,
          type: 'text',
          sender: 'bot',
          timestamp: new Date().toISOString(),
        }
        setMessages(prev => [...prev, botMessage])
      }, 1000)
    } else {
      try {
        // Mock send message for demo
        console.log('Sending message:', message.trim())
        
        // Add mock bot response
        setTimeout(() => {
          const botMessage: ChatMessage = {
            id: (Date.now() + 1).toString(),
            content: `Tôi hiểu bạn đang hỏi về "${message.trim()}". Đây là câu trả lời mẫu. Để xem biểu đồ, bạn có thể gõ "biểu đồ" hoặc "chart".`,
            type: 'text',
            sender: 'bot',
            timestamp: new Date().toISOString(),
          }
          setMessages(prev => [...prev, botMessage])
        }, 1000)
      } catch (error) {
        console.error('Failed to send message:', error)
      }
    }
  }

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSendMessage()
    }
  }

  const formatTime = (timestamp: string) => {
    return new Date(timestamp).toLocaleTimeString('vi-VN', {
      hour: '2-digit',
      minute: '2-digit',
    })
  }

  const renderMessage = (msg: ChatMessage) => {
    const isUser = msg.sender === 'user'
    
    return (
      <ListItem
        key={msg.id}
        sx={{
          display: 'flex',
          justifyContent: isUser ? 'flex-end' : 'flex-start',
          alignItems: 'flex-start',
          mb: 2,
        }}
      >
        <Box
          sx={{
            display: 'flex',
            flexDirection: isUser ? 'row-reverse' : 'row',
            alignItems: 'flex-start',
            maxWidth: '70%',
            gap: 1,
          }}
        >
          <Avatar
            sx={{
              bgcolor: isUser ? 'primary.main' : 'secondary.main',
              width: 32,
              height: 32,
            }}
          >
            {isUser ? <PersonIcon /> : <BotIcon />}
          </Avatar>
          
          <Box
            sx={{
              bgcolor: isUser ? 'primary.main' : 'grey.100',
              color: isUser ? 'primary.contrastText' : 'text.primary',
              p: 2,
              borderRadius: 2,
              position: 'relative',
            }}
          >
            <Typography variant="body1" sx={{ wordBreak: 'break-word' }}>
              {msg.content}
            </Typography>
            
            <Typography
              variant="caption"
              sx={{
                display: 'block',
                mt: 1,
                opacity: 0.7,
                fontSize: '0.75rem',
              }}
            >
              {formatTime(msg.timestamp)}
            </Typography>
          </Box>
        </Box>
      </ListItem>
    )
  }

  return (
    <Box sx={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
      {/* Header */}
      <Paper sx={{ p: 2, mb: 2 }}>
        <Typography variant="h5" component="h1" gutterBottom>
          Chat với AI
        </Typography>
        <Typography variant="body2" color="text.secondary">
          Hỏi bất kỳ câu hỏi nào về cổ phiếu và đầu tư. Gõ "biểu đồ" để xem chart!
        </Typography>
      </Paper>

      {/* Messages */}
      <Paper
        sx={{
          flexGrow: 1,
          display: 'flex',
          flexDirection: 'column',
          overflow: 'hidden',
          width: showChart ? '50%' : '100%',
          transition: 'width 0.3s ease',
        }}
      >
        <Box
          sx={{
            flexGrow: 1,
            overflow: 'auto',
            p: 2,
            minHeight: 400,
          }}
        >
          {isLoading && messages.length === 0 ? (
            <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '100%' }}>
              <LoadingSpinner message="Đang tải cuộc trò chuyện..." />
            </Box>
          ) : messages.length === 0 ? (
            <Box sx={{ textAlign: 'center', py: 4 }}>
              <Typography variant="h6" color="text.secondary" gutterBottom>
                Chào mừng bạn đến với Stock Chat Bot!
              </Typography>
              <Typography variant="body2" color="text.secondary">
                Hãy bắt đầu cuộc trò chuyện bằng cách hỏi về cổ phiếu, phân tích thị trường, hoặc bất kỳ câu hỏi nào về đầu tư.
              </Typography>
            </Box>
          ) : (
            <List sx={{ p: 0 }}>
              {messages.map(renderMessage)}
              {isTyping && (
                <ListItem sx={{ justifyContent: 'flex-start' }}>
                  <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                    <Avatar sx={{ bgcolor: 'secondary.main', width: 32, height: 32 }}>
                      <BotIcon />
                    </Avatar>
                    <Chip label="AI đang trả lời..." size="small" />
                  </Box>
                </ListItem>
              )}
            </List>
          )}
        </Box>

        <Divider />

        {/* Message Input */}
        <Box sx={{ p: 2 }}>
          <Box sx={{ display: 'flex', gap: 1, alignItems: 'flex-end' }}>
            <TextField
              fullWidth
              multiline
              maxRows={4}
              placeholder="Nhập câu hỏi của bạn..."
              value={message}
              onChange={(e) => setMessage(e.target.value)}
              onKeyPress={handleKeyPress}
              disabled={isLoading}
              variant="outlined"
              size="small"
            />
            <IconButton
              color="primary"
              onClick={handleSendMessage}
              disabled={!message.trim() || isLoading}
              sx={{ alignSelf: 'flex-end' }}
            >
              <SendIcon />
            </IconButton>
          </Box>
        </Box>
      </Paper>

      {/* Chart Viewport */}
      <ChartViewport
        isVisible={showChart}
        onClose={() => setShowChart(false)}
        symbol={chartSymbol}
        data={[]}
      />
    </Box>
  )
}

export default ChatPage
