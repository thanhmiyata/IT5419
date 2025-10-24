import { useCallback } from 'react'
import { useDispatch, useSelector } from 'react-redux'

import { RootState, AppDispatch } from '@/store'
import { addNotification, removeNotification, markNotificationAsRead } from '@/store/slices/uiSlice'

export const useNotifications = () => {
  const dispatch = useDispatch<AppDispatch>()
  const { notifications } = useSelector((state: RootState) => state.ui)

  const showNotification = useCallback((
    message: string,
    type: 'success' | 'error' | 'warning' | 'info' = 'info'
  ) => {
    dispatch(addNotification({
      type,
      message,
    }))
  }, [dispatch])

  const showSuccess = useCallback((message: string) => {
    showNotification(message, 'success')
  }, [showNotification])

  const showError = useCallback((message: string) => {
    showNotification(message, 'error')
  }, [showNotification])

  const showWarning = useCallback((message: string) => {
    showNotification(message, 'warning')
  }, [showNotification])

  const showInfo = useCallback((message: string) => {
    showNotification(message, 'info')
  }, [showNotification])

  const removeNotificationById = useCallback((id: string) => {
    dispatch(removeNotification(id))
  }, [dispatch])

  const markAsRead = useCallback((id: string) => {
    dispatch(markNotificationAsRead(id))
  }, [dispatch])

  const unreadCount = notifications.filter(n => !n.read).length

  return {
    notifications,
    unreadCount,
    showNotification,
    showSuccess,
    showError,
    showWarning,
    showInfo,
    removeNotification: removeNotificationById,
    markAsRead,
  }
}
