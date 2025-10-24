import { createSlice, createAsyncThunk, PayloadAction } from '@reduxjs/toolkit'
import { ChatState, ChatMessage, ChatSession } from '@/types'
import { chatService } from '@/services/chatService'

const initialState: ChatState = {
  currentSession: null,
  sessions: [],
  isLoading: false,
  isTyping: false,
  error: null,
}

// Async thunks
export const createChatSession = createAsyncThunk(
  'chat/createSession',
  async (title: string, { rejectWithValue }) => {
    try {
      const response = await chatService.createSession(title)
      return response
    } catch (error: any) {
      return rejectWithValue(error.message || 'Failed to create session')
    }
  }
)

export const getChatSessions = createAsyncThunk(
  'chat/getSessions',
  async (_, { rejectWithValue }) => {
    try {
      const response = await chatService.getSessions()
      return response
    } catch (error: any) {
      return rejectWithValue(error.message || 'Failed to get sessions')
    }
  }
)

export const sendMessage = createAsyncThunk(
  'chat/sendMessage',
  async ({ sessionId, content }: { sessionId: string; content: string }, { rejectWithValue }) => {
    try {
      const response = await chatService.sendMessage(sessionId, content)
      return response
    } catch (error: any) {
      return rejectWithValue(error.message || 'Failed to send message')
    }
  }
)

export const getSessionMessages = createAsyncThunk(
  'chat/getSessionMessages',
  async (sessionId: string, { rejectWithValue }) => {
    try {
      const response = await chatService.getSessionMessages(sessionId)
      return response
    } catch (error: any) {
      return rejectWithValue(error.message || 'Failed to get messages')
    }
  }
)

const chatSlice = createSlice({
  name: 'chat',
  initialState,
  reducers: {
    setCurrentSession: (state, action: PayloadAction<ChatSession | null>) => {
      state.currentSession = action.payload
    },
    addMessage: (state, action: PayloadAction<ChatMessage>) => {
      if (state.currentSession) {
        state.currentSession.messages.push(action.payload)
      }
    },
    setTyping: (state, action: PayloadAction<boolean>) => {
      state.isTyping = action.payload
    },
    clearError: (state) => {
      state.error = null
    },
    updateMessage: (state, action: PayloadAction<{ messageId: string; updates: Partial<ChatMessage> }>) => {
      if (state.currentSession) {
        const messageIndex = state.currentSession.messages.findIndex(
          msg => msg.id === action.payload.messageId
        )
        if (messageIndex !== -1) {
          state.currentSession.messages[messageIndex] = {
            ...state.currentSession.messages[messageIndex],
            ...action.payload.updates
          }
        }
      }
    },
    clearCurrentSession: (state) => {
      state.currentSession = null
    },
  },
  extraReducers: (builder) => {
    builder
      // Create session
      .addCase(createChatSession.pending, (state) => {
        state.isLoading = true
        state.error = null
      })
      .addCase(createChatSession.fulfilled, (state, action) => {
        state.isLoading = false
        state.sessions.unshift(action.payload)
        state.currentSession = action.payload
        state.error = null
      })
      .addCase(createChatSession.rejected, (state, action) => {
        state.isLoading = false
        state.error = action.payload as string
      })
      // Get sessions
      .addCase(getChatSessions.pending, (state) => {
        state.isLoading = true
      })
      .addCase(getChatSessions.fulfilled, (state, action) => {
        state.isLoading = false
        state.sessions = action.payload
        state.error = null
      })
      .addCase(getChatSessions.rejected, (state, action) => {
        state.isLoading = false
        state.error = action.payload as string
      })
      // Send message
      .addCase(sendMessage.pending, (state) => {
        state.isLoading = true
        state.error = null
      })
      .addCase(sendMessage.fulfilled, (state, action) => {
        state.isLoading = false
        if (state.currentSession) {
          state.currentSession.messages.push(action.payload)
        }
        state.error = null
      })
      .addCase(sendMessage.rejected, (state, action) => {
        state.isLoading = false
        state.error = action.payload as string
      })
      // Get session messages
      .addCase(getSessionMessages.pending, (state) => {
        state.isLoading = true
      })
      .addCase(getSessionMessages.fulfilled, (state, action) => {
        state.isLoading = false
        if (state.currentSession) {
          state.currentSession.messages = action.payload
        }
        state.error = null
      })
      .addCase(getSessionMessages.rejected, (state, action) => {
        state.isLoading = false
        state.error = action.payload as string
      })
  },
})

export const {
  setCurrentSession,
  addMessage,
  setTyping,
  clearError,
  updateMessage,
  clearCurrentSession,
} = chatSlice.actions

export default chatSlice.reducer
