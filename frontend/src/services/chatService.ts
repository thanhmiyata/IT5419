import { apiService } from './api'
import { ChatSession, ChatMessage, ApiResponse } from '@/types'

class ChatService {
  async createSession(title: string): Promise<ChatSession> {
    const response = await apiService.post<ChatSession>('/chat/sessions', { title })
    if (!response.success) {
      throw new Error(response.error || 'Failed to create session')
    }
    return response.data
  }

  async getSessions(): Promise<ChatSession[]> {
    const response = await apiService.get<ChatSession[]>('/chat/sessions')
    if (!response.success) {
      throw new Error(response.error || 'Failed to get sessions')
    }
    return response.data
  }

  async getSession(sessionId: string): Promise<ChatSession> {
    const response = await apiService.get<ChatSession>(`/chat/sessions/${sessionId}`)
    if (!response.success) {
      throw new Error(response.error || 'Failed to get session')
    }
    return response.data
  }

  async updateSession(sessionId: string, updates: Partial<ChatSession>): Promise<ChatSession> {
    const response = await apiService.put<ChatSession>(`/chat/sessions/${sessionId}`, updates)
    if (!response.success) {
      throw new Error(response.error || 'Failed to update session')
    }
    return response.data
  }

  async deleteSession(sessionId: string): Promise<void> {
    const response = await apiService.delete(`/chat/sessions/${sessionId}`)
    if (!response.success) {
      throw new Error(response.error || 'Failed to delete session')
    }
  }

  async getSessionMessages(sessionId: string): Promise<ChatMessage[]> {
    const response = await apiService.get<ChatMessage[]>(`/chat/sessions/${sessionId}/messages`)
    if (!response.success) {
      throw new Error(response.error || 'Failed to get messages')
    }
    return response.data
  }

  async sendMessage(sessionId: string, content: string): Promise<ChatMessage> {
    const response = await apiService.post<ChatMessage>(`/chat/sessions/${sessionId}/messages`, {
      content,
    })
    if (!response.success) {
      throw new Error(response.error || 'Failed to send message')
    }
    return response.data
  }

  async deleteMessage(sessionId: string, messageId: string): Promise<void> {
    const response = await apiService.delete(`/chat/sessions/${sessionId}/messages/${messageId}`)
    if (!response.success) {
      throw new Error(response.error || 'Failed to delete message')
    }
  }

  async updateMessage(
    sessionId: string,
    messageId: string,
    updates: Partial<ChatMessage>
  ): Promise<ChatMessage> {
    const response = await apiService.put<ChatMessage>(
      `/chat/sessions/${sessionId}/messages/${messageId}`,
      updates
    )
    if (!response.success) {
      throw new Error(response.error || 'Failed to update message')
    }
    return response.data
  }

  async exportSession(sessionId: string, format: 'pdf' | 'json' | 'txt'): Promise<Blob> {
    const response = await apiService.get(`/chat/sessions/${sessionId}/export?format=${format}`, {
      responseType: 'blob',
    })
    return response.data
  }

  async searchMessages(query: string, sessionId?: string): Promise<ChatMessage[]> {
    const params = new URLSearchParams({ q: query })
    if (sessionId) {
      params.append('sessionId', sessionId)
    }
    
    const response = await apiService.get<ChatMessage[]>(`/chat/search?${params}`)
    if (!response.success) {
      throw new Error(response.error || 'Failed to search messages')
    }
    return response.data
  }
}

export const chatService = new ChatService()
