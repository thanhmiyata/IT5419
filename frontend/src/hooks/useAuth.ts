import { useSelector, useDispatch } from 'react-redux'
import { useNavigate } from 'react-router-dom'
import { useCallback } from 'react'

import { RootState, AppDispatch } from '@/store'
import { logoutUser } from '@/store/slices/authSlice'

export const useAuth = () => {
  const dispatch = useDispatch<AppDispatch>()
  const navigate = useNavigate()
  const { user, isAuthenticated, isLoading, error } = useSelector((state: RootState) => state.auth)

  const logout = useCallback(async () => {
    try {
      await dispatch(logoutUser()).unwrap()
      navigate('/login')
    } catch (error) {
      console.error('Logout failed:', error)
    }
  }, [dispatch, navigate])

  const isAdmin = user?.role === 'admin'
  const isUser = user?.role === 'user'

  return {
    user,
    isAuthenticated,
    isLoading,
    error,
    logout,
    isAdmin,
    isUser,
  }
}
