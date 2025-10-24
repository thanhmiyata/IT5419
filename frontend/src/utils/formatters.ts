// Currency formatting
export const formatCurrency = (value: number, currency: string = 'VND'): string => {
  return new Intl.NumberFormat('vi-VN', {
    style: 'currency',
    currency,
  }).format(value)
}

// Number formatting
export const formatNumber = (value: number): string => {
  return new Intl.NumberFormat('vi-VN').format(value)
}

// Percentage formatting
export const formatPercentage = (value: number, decimals: number = 2): string => {
  return `${value >= 0 ? '+' : ''}${value.toFixed(decimals)}%`
}

// Date formatting
export const formatDate = (date: string | Date, format: 'short' | 'long' | 'time' = 'short'): string => {
  const dateObj = typeof date === 'string' ? new Date(date) : date
  
  switch (format) {
    case 'short':
      return dateObj.toLocaleDateString('vi-VN')
    case 'long':
      return dateObj.toLocaleDateString('vi-VN', {
        year: 'numeric',
        month: 'long',
        day: 'numeric',
      })
    case 'time':
      return dateObj.toLocaleTimeString('vi-VN', {
        hour: '2-digit',
        minute: '2-digit',
      })
    default:
      return dateObj.toLocaleDateString('vi-VN')
  }
}

// File size formatting
export const formatFileSize = (bytes: number): string => {
  if (bytes === 0) return '0 Bytes'
  
  const k = 1024
  const sizes = ['Bytes', 'KB', 'MB', 'GB', 'TB']
  const i = Math.floor(Math.log(bytes) / Math.log(k))
  
  return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i]
}

// Stock symbol formatting
export const formatStockSymbol = (symbol: string): string => {
  return symbol.toUpperCase().trim()
}

// Price change formatting
export const formatPriceChange = (change: number, changePercent: number): string => {
  const sign = change >= 0 ? '+' : ''
  return `${sign}${formatCurrency(change)} (${formatPercentage(changePercent)})`
}

// Volume formatting
export const formatVolume = (volume: number): string => {
  if (volume >= 1000000000) {
    return (volume / 1000000000).toFixed(1) + 'B'
  } else if (volume >= 1000000) {
    return (volume / 1000000).toFixed(1) + 'M'
  } else if (volume >= 1000) {
    return (volume / 1000).toFixed(1) + 'K'
  }
  return volume.toString()
}

// Market cap formatting
export const formatMarketCap = (marketCap: number): string => {
  if (marketCap >= 1000000000000) {
    return (marketCap / 1000000000000).toFixed(1) + 'T'
  } else if (marketCap >= 1000000000) {
    return (marketCap / 1000000000).toFixed(1) + 'B'
  } else if (marketCap >= 1000000) {
    return (marketCap / 1000000).toFixed(1) + 'M'
  }
  return formatCurrency(marketCap)
}

// Time ago formatting
export const formatTimeAgo = (date: string | Date): string => {
  const now = new Date()
  const past = typeof date === 'string' ? new Date(date) : date
  const diffInSeconds = Math.floor((now.getTime() - past.getTime()) / 1000)
  
  if (diffInSeconds < 60) {
    return 'Vừa xong'
  } else if (diffInSeconds < 3600) {
    const minutes = Math.floor(diffInSeconds / 60)
    return `${minutes} phút trước`
  } else if (diffInSeconds < 86400) {
    const hours = Math.floor(diffInSeconds / 3600)
    return `${hours} giờ trước`
  } else if (diffInSeconds < 2592000) {
    const days = Math.floor(diffInSeconds / 86400)
    return `${days} ngày trước`
  } else {
    return formatDate(past, 'short')
  }
}
