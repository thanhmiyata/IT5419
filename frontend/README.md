# Stock Chat Bot Frontend

Frontend cho hệ thống Bot chat tư vấn cổ phiếu thông minh với AI.

## 🚀 Tính năng

- **Chat Interface**: Giao diện chat trực quan với AI
- **Stock Analysis**: Phân tích cổ phiếu và thị trường
- **Portfolio Management**: Quản lý danh mục đầu tư
- **Real-time Data**: Dữ liệu thời gian thực qua WebSocket
- **Responsive Design**: Giao diện thân thiện trên mọi thiết bị

## 🛠️ Công nghệ sử dụng

- **React 18** với TypeScript
- **Material-UI (MUI)** cho UI components
- **Redux Toolkit** cho state management
- **React Router** cho routing
- **Chart.js** cho biểu đồ
- **Socket.io** cho WebSocket
- **Vite** cho build tool

## 📦 Cài đặt

1. **Clone repository**
```bash
git clone <repository-url>
cd frontend
```

2. **Cài đặt dependencies**
```bash
npm install
```

3. **Cấu hình environment**
```bash
cp env.example .env
# Chỉnh sửa các biến trong .env theo môi trường của bạn
```

4. **Chạy development server**
```bash
npm run dev
```

## 🏗️ Cấu trúc thư mục

```
src/
├── components/          # React components
│   ├── Auth/           # Authentication components
│   ├── Layout/         # Layout components
│   └── UI/             # Reusable UI components
├── pages/              # Page components
├── services/           # API services
├── store/              # Redux store và slices
├── types/              # TypeScript type definitions
├── utils/              # Utility functions
└── hooks/              # Custom React hooks
```

## 🔧 Scripts

- `npm run dev` - Chạy development server
- `npm run build` - Build production
- `npm run preview` - Preview production build
- `npm run lint` - Chạy ESLint
- `npm run lint:fix` - Fix ESLint errors

## 🌐 API Integration

Frontend tích hợp với backend API qua:

- **REST API**: Cho các operations CRUD
- **WebSocket**: Cho real-time communication
- **Authentication**: JWT token-based

## 📱 Responsive Design

- **Mobile-first**: Thiết kế ưu tiên mobile
- **Breakpoints**: 
  - xs: 0px+
  - sm: 600px+
  - md: 900px+
  - lg: 1200px+
  - xl: 1536px+

## 🎨 Theme & Styling

- **Material-UI Theme**: Custom theme với màu sắc thương hiệu
- **CSS-in-JS**: Styled components với emotion
- **Dark/Light Mode**: Hỗ trợ chế độ sáng/tối

## 🔐 Authentication

- **JWT Tokens**: Lưu trữ trong localStorage
- **Protected Routes**: Bảo vệ các route cần authentication
- **Auto Refresh**: Tự động refresh token khi hết hạn

## 📊 State Management

Sử dụng Redux Toolkit với các slices:

- **authSlice**: Quản lý authentication state
- **chatSlice**: Quản lý chat sessions và messages
- **stockSlice**: Quản lý stock data và analysis
- **uiSlice**: Quản lý UI state (modals, notifications, etc.)

## 🚀 Deployment

### Production Build

```bash
npm run build
```

### Environment Variables

Cần cấu hình các biến môi trường sau:

- `VITE_API_URL`: URL của backend API
- `VITE_WS_URL`: URL của WebSocket server
- `VITE_APP_NAME`: Tên ứng dụng
- `VITE_APP_VERSION`: Phiên bản ứng dụng

### Docker

```dockerfile
FROM node:18-alpine
WORKDIR /app
COPY package*.json ./
RUN npm ci --only=production
COPY . .
RUN npm run build
EXPOSE 3000
CMD ["npm", "run", "preview"]
```

## 🧪 Testing

```bash
# Chạy tests
npm test

# Chạy tests với coverage
npm run test:coverage

# Chạy E2E tests
npm run test:e2e
```

## 📈 Performance

- **Code Splitting**: Lazy loading cho các routes
- **Bundle Optimization**: Tree shaking và minification
- **Caching**: Service worker cho offline support
- **Image Optimization**: Lazy loading và compression

## 🔍 Monitoring

- **Error Tracking**: Sentry integration
- **Analytics**: Google Analytics
- **Performance**: Web Vitals monitoring

## 🤝 Contributing

1. Fork repository
2. Tạo feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Tạo Pull Request

## 📄 License

Dự án này được phân phối dưới giấy phép MIT. Xem file `LICENSE` để biết thêm chi tiết.

## 📞 Support

Nếu bạn gặp vấn đề hoặc có câu hỏi, vui lòng:

- Tạo issue trên GitHub
- Liên hệ team development
- Xem documentation chi tiết

---

**Stock Chat Bot** - Hệ thống tư vấn cổ phiếu thông minh với AI
