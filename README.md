# 🛡️ DeepShield - AI Deepfake Detector

A modern, web-based deepfake detection system with enhanced UI/UX design.

## 🚀 Quick Start

### Method 1: Using the Batch File (Windows)
```bash
start_server.bat
```

### Method 2: Using Python Directly
```bash
python simple_server.py
```

### Method 3: Using npm
```bash
npm run dev
```

Then open your browser and navigate to:
- **Frontend**: http://localhost:8080
- **API Health**: http://localhost:8080/api/health

## 🎨 UI/UX Features

- **Modern Glassmorphism Design** with enhanced visual effects
- **Smooth Animations** and micro-interactions
- **Responsive Layout** that works on all devices
- **Interactive Components** with hover effects and transitions
- **Gradient Text** and modern typography
- **Animated Background** with floating orbs

## 🤖 Model Training

### Simple Training (Recommended for quick testing)
```bash
python simple_train.py
```

### Advanced Training (with real datasets)
```bash
python run_training.py --dataset cifake --epochs 10 --batch_size 8
```

## 📁 Project Structure

```
deepfake-detector/
├── frontend/           # Enhanced UI with modern design
│   ├── index.html      # Main application interface
│   ├── styles.css      # Modern CSS with animations
│   └── app.js          # Frontend JavaScript
├── backend/            # Detection models and API
│   ├── training/       # Training scripts
│   ├── inference/      # Model inference
│   └── api/           # FastAPI endpoints
├── data_pipeline/      # Dataset download and processing
├── simple_server.py    # Development server
└── simple_train.py     # Simplified training script
```

## 🔧 Features

- **Multi-Modal Detection**: Images, Videos, and Audio
- **Real-time Analysis**: Fast inference with visual feedback
- **Grad-CAM Heatmaps**: Visual explanations of detection results
- **Modality Breakdown**: Detailed analysis across different detection methods
- **Modern UI**: Beautiful, responsive interface with smooth animations

## 🛠️ Technologies Used

- **Frontend**: HTML5, CSS3, JavaScript (ES6+)
- **Backend**: Python, FastAPI, PyTorch
- **ML Models**: EfficientNet-B4, TimeSformer, wav2vec2
- **Design**: Glassmorphism, CSS Grid, Flexbox, CSS Animations

## 📊 API Endpoints

- `GET /api/health` - Health check
- `POST /api/analyze` - Analyze media for deepfakes
- `GET /styles.css` - CSS styles
- `GET /app.js` - Frontend JavaScript

## 🎯 Key Improvements Made

1. **Enhanced Visual Design**: Modern gradients, better contrast, improved typography
2. **Smooth Animations**: Cubic-bezier transitions, hover effects, micro-interactions
3. **Better UX**: Interactive components, visual feedback, responsive design
4. **Performance**: Optimized CSS, efficient animations, fast loading
5. **Accessibility**: Semantic HTML, proper ARIA labels, keyboard navigation

## 🌟 Demo Features

The demo includes:
- **Mock Analysis**: Shows how the detection results would look
- **Interactive UI**: Test all the animations and interactions
- **Responsive Design**: Works on desktop, tablet, and mobile
- **Modern Effects**: Glassmorphism, gradients, animations

---

🛡️ **DeepShield** - Protecting digital authenticity with AI-powered detection
