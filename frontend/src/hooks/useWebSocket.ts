import { useEffect, useCallback } from 'react'
import { useDispatch } from 'react-redux'
import { useSelector } from 'react-redux'

import { RootState, AppDispatch } from '@/store'
import { websocketService } from '@/services/websocketService'
import { addMessage, setTyping } from '@/store/slices/chatSlice'

export const useWebSocket = () => {
  const dispatch = useDispatch<AppDispatch>()
  const { isAuthenticated, token } = useSelector((state: RootState) => state.auth)
  const { currentSession } = useSelector((state: RootState) => state.chat)

  const connect = useCallback(async () => {
    if (isAuthenticated && token) {
      try {
        await websocketService.connect(token)
        console.log('WebSocket connected')
      } catch (error) {
        console.error('WebSocket connection failed:', error)
      }
    }
  }, [isAuthenticated, token])

  const disconnect = useCallback(() => {
    websocketService.disconnect()
  }, [])

  const sendMessage = useCallback((content: string) => {
    if (currentSession) {
      websocketService.sendChatMessage(currentSession.id, content)
    }
  }, [currentSession])

  const joinSession = useCallback((sessionId: string) => {
    websocketService.joinChatSession(sessionId)
  }, [])

  const leaveSession = useCallback((sessionId: string) => {
    websocketService.leaveChatSession(sessionId)
  }, [])

  useEffect(() => {
    // Setup event listeners
    const handleChatMessage = (message: any) => {
      dispatch(addMessage(message))
    }

    const handleTyping = (data: any) => {
      if (data.sessionId === currentSession?.id) {
        dispatch(setTyping(data.isTyping))
      }
    }

    // Register event listeners
    websocketService.onChatMessage(handleChatMessage)
    websocketService.onTyping(handleTyping)

    // Connect when authenticated
    if (isAuthenticated) {
      connect()
    }

    // Cleanup
    return () => {
      websocketService.off('chat_message', handleChatMessage)
      websocketService.off('typing', handleTyping)
      disconnect()
    }
  }, [isAuthenticated, connect, disconnect, dispatch, currentSession])

  return {
    isConnected: websocketService.isConnected(),
    sendMessage,
    joinSession,
    leaveSession,
    connect,
    disconnect,
  }
}
