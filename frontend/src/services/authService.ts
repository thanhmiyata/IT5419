import { apiService } from './api'
import { User, LoginRequest, RegisterRequest } from '@/types'

interface AuthResponse {
  user: User
  token: string
}

class AuthService {
  async login(credentials: LoginRequest): Promise<AuthResponse> {
    const response = await apiService.post<AuthResponse>('/auth/login', credentials)
    if (!response.success) {
      throw new Error(response.error || 'Login failed')
    }
    return response.data
  }

  async register(userData: RegisterRequest): Promise<AuthResponse> {
    const response = await apiService.post<AuthResponse>('/auth/register', userData)
    if (!response.success) {
      throw new Error(response.error || 'Registration failed')
    }
    return response.data
  }

  async logout(): Promise<void> {
    await apiService.post('/auth/logout')
  }

  async getCurrentUser(): Promise<User> {
    const response = await apiService.get<User>('/auth/me')
    if (!response.success) {
      throw new Error(response.error || 'Failed to get user info')
    }
    return response.data
  }

  async refreshToken(): Promise<AuthResponse> {
    const response = await apiService.post<AuthResponse>('/auth/refresh')
    if (!response.success) {
      throw new Error(response.error || 'Token refresh failed')
    }
    return response.data
  }

  async forgotPassword(email: string): Promise<void> {
    const response = await apiService.post('/auth/forgot-password', { email })
    if (!response.success) {
      throw new Error(response.error || 'Failed to send reset email')
    }
  }

  async resetPassword(token: string, newPassword: string): Promise<void> {
    const response = await apiService.post('/auth/reset-password', {
      token,
      password: newPassword,
    })
    if (!response.success) {
      throw new Error(response.error || 'Failed to reset password')
    }
  }

  async changePassword(currentPassword: string, newPassword: string): Promise<void> {
    const response = await apiService.post('/auth/change-password', {
      currentPassword,
      newPassword,
    })
    if (!response.success) {
      throw new Error(response.error || 'Failed to change password')
    }
  }

  async updateProfile(userData: Partial<User>): Promise<User> {
    const response = await apiService.put<User>('/auth/profile', userData)
    if (!response.success) {
      throw new Error(response.error || 'Failed to update profile')
    }
    return response.data
  }
}

export const authService = new AuthService()
