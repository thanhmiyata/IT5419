import { io, Socket } from 'socket.io-client'
import { WebSocketMessage, ChatMessage } from '@/types'

class WebSocketService {
  private socket: Socket | null = null
  private reconnectAttempts = 0
  private maxReconnectAttempts = 5
  private reconnectDelay = 1000

  connect(token?: string): Promise<void> {
    return new Promise((resolve, reject) => {
      const wsUrl = import.meta.env.VITE_WS_URL || 'ws://localhost:8000'
      
      this.socket = io(wsUrl, {
        auth: {
          token: token || localStorage.getItem('token'),
        },
        transports: ['websocket'],
        timeout: 10000,
      })

      this.socket.on('connect', () => {
        console.log('WebSocket connected')
        this.reconnectAttempts = 0
        resolve()
      })

      this.socket.on('connect_error', (error) => {
        console.error('WebSocket connection error:', error)
        reject(error)
      })

      this.socket.on('disconnect', (reason) => {
        console.log('WebSocket disconnected:', reason)
        if (reason === 'io server disconnect') {
          // Server disconnected, try to reconnect
          this.handleReconnect()
        }
      })

      this.socket.on('reconnect', (attemptNumber) => {
        console.log('WebSocket reconnected after', attemptNumber, 'attempts')
        this.reconnectAttempts = 0
      })

      this.socket.on('reconnect_error', (error) => {
        console.error('WebSocket reconnect error:', error)
        this.handleReconnect()
      })

      this.socket.on('reconnect_failed', () => {
        console.error('WebSocket reconnection failed')
        this.reconnectAttempts = this.maxReconnectAttempts
      })
    })
  }

  private handleReconnect() {
    if (this.reconnectAttempts < this.maxReconnectAttempts) {
      this.reconnectAttempts++
      const delay = this.reconnectDelay * Math.pow(2, this.reconnectAttempts - 1)
      console.log(`Attempting to reconnect in ${delay}ms (attempt ${this.reconnectAttempts})`)
      
      setTimeout(() => {
        if (this.socket) {
          this.socket.connect()
        }
      }, delay)
    }
  }

  disconnect(): void {
    if (this.socket) {
      this.socket.disconnect()
      this.socket = null
    }
  }

  isConnected(): boolean {
    return this.socket?.connected || false
  }

  // Chat related methods
  joinChatSession(sessionId: string): void {
    if (this.socket) {
      this.socket.emit('join_chat_session', { sessionId })
    }
  }

  leaveChatSession(sessionId: string): void {
    if (this.socket) {
      this.socket.emit('leave_chat_session', { sessionId })
    }
  }

  sendChatMessage(sessionId: string, content: string): void {
    if (this.socket) {
      this.socket.emit('chat_message', { sessionId, content })
    }
  }

  // Event listeners
  onChatMessage(callback: (message: ChatMessage) => void): void {
    if (this.socket) {
      this.socket.on('chat_message', callback)
    }
  }

  onTyping(callback: (data: { sessionId: string; isTyping: boolean; userId: string }) => void): void {
    if (this.socket) {
      this.socket.on('typing', callback)
    }
  }

  onUserJoined(callback: (data: { sessionId: string; userId: string }) => void): void {
    if (this.socket) {
      this.socket.on('user_joined', callback)
    }
  }

  onUserLeft(callback: (data: { sessionId: string; userId: string }) => void): void {
    if (this.socket) {
      this.socket.on('user_left', callback)
    }
  }

  // Stock data related methods
  subscribeToStock(symbol: string): void {
    if (this.socket) {
      this.socket.emit('subscribe_stock', { symbol })
    }
  }

  unsubscribeFromStock(symbol: string): void {
    if (this.socket) {
      this.socket.emit('unsubscribe_stock', { symbol })
    }
  }

  onStockUpdate(callback: (data: any) => void): void {
    if (this.socket) {
      this.socket.on('stock_update', callback)
    }
  }

  onMarketUpdate(callback: (data: any) => void): void {
    if (this.socket) {
      this.socket.on('market_update', callback)
    }
  }

  // Generic event handling
  on(event: string, callback: (...args: any[]) => void): void {
    if (this.socket) {
      this.socket.on(event, callback)
    }
  }

  off(event: string, callback?: (...args: any[]) => void): void {
    if (this.socket) {
      if (callback) {
        this.socket.off(event, callback)
      } else {
        this.socket.off(event)
      }
    }
  }

  emit(event: string, data?: any): void {
    if (this.socket) {
      this.socket.emit(event, data)
    }
  }

  // Error handling
  onError(callback: (error: any) => void): void {
    if (this.socket) {
      this.socket.on('error', callback)
    }
  }

  // Connection status
  onConnectionStatus(callback: (status: 'connected' | 'disconnected' | 'connecting') => void): void {
    if (this.socket) {
      this.socket.on('connect', () => callback('connected'))
      this.socket.on('disconnect', () => callback('disconnected'))
      this.socket.on('reconnect', () => callback('connected'))
    }
  }
}

export const websocketService = new WebSocketService()
