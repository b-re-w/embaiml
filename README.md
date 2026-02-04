# Embedded AI and Machine Learning Course

**University of Nevada, Las Vegas (UNLV)**  
**February 2026**

## 📋 Course Overview

This intensive program explores the integration of artificial intelligence and machine learning techniques into embedded systems. Students will learn to develop intelligent, efficient, and cost-effective IoT devices on resource-constrained platforms including microcontrollers and single-board computers.

## 🎯 Learning Objectives

- Understand the constraints and applications of Edge and Embedded ML
- Master dual-platform development (MCU vs SBC)
- Build end-to-end ML pipelines from data collection to deployment
- Optimize models for resource-constrained devices
- Implement secure and efficient edge-cloud collaboration systems

## 🛠️ Hardware Requirements

### Core Components (Per Student/Team)
- **M5Stack Core2** - ESP32-based development kit with touchscreen, microphone, IMU, Wi-Fi/BLE
- **Raspberry Pi Zero 2 W** - with case, power supply, 32GB microSD card
- **Raspberry Pi Camera Module v2** - for vision-based projects
- USB-C and micro-HDMI cables

### Optional Add-ons
- BME688 air quality sensor
- MAX9814 microphone module
- Thermal camera module
- Portable battery pack

## 💻 Software & Tools

### M5Core2 Development
- Arduino IDE / PlatformIO
- M5Stack UIFlow (optional)
- TensorFlow Lite Micro
- Edge Impulse SDK

### Raspberry Pi Development
- Raspberry Pi OS Lite
- Python 3.x
- TensorFlow Lite
- OpenCV
- MQTT
- scikit-learn

### Model Optimization
- TensorFlow Model Optimization Toolkit
- Post-training quantization tools

### Communication & Analysis
- MQTT broker
- Python, Pandas, Matplotlib

## 📅 Course Schedule

| Day | Lecture Topic | Lab Activity | Tools |
|-----|--------------|--------------|-------|
| 1 | Intro to Edge & Embedded ML | Setup M5Core2 & Pi Zero 2, "Hello World" | Arduino IDE, Raspberry Pi OS |
| 2 | MCU vs SBC Architecture | Read IMU & mic data from M5Core2 | Arduino IDE, M5Stack libraries |
| 3 | Sensor Data Acquisition | Log data to SD card & send to Pi over Wi-Fi | Arduino IDE, MQTT, Python |
| 4 | Data Preprocessing & Feature Extraction | Extract IMU features, compare preprocessing | Arduino IDE, Python |
| 5 | ML Fundamentals for Embedded Devices | Train MLP classifier on Pi | scikit-learn, matplotlib |
| 6 | Deploying Models to Microcontrollers | Convert & deploy TFLite Micro model | TFLite Micro, Arduino IDE |
| 7 | Model Optimization | Quantize model, redeploy to M5Core2 | TF Model Optimization Toolkit |
| 8 | Audio Keyword Spotting | Capture audio, train, deploy | Edge Impulse Studio |
| 9 | Vision-based ML on Pi | Run MobileNet with Pi Camera | TFLite, OpenCV |
| 10 | Wireless ML Inference Pipelines | Send inference results via MQTT dashboard | Arduino IDE, Python MQTT |
| 11 | Edge-Cloud Collaboration | Split inference pipeline | MQTT, TFLite |
| 12 | Security in Edge ML | Secure OTA firmware updates | M5Burner, HTTPS |
| 13 | Multi-Device ML | Deploy same model, compare performance | TFLite, Arduino IDE |
| 14 | Project Development | Build end-to-end applications | All tools |
| 15 | Final Project Demos | Present and test applications | All tools |

## 🚀 Example Final Projects

### 1. Gesture-Controlled Presentation Pointer
Use M5Core2's IMU to detect hand gestures controlling slides via Pi Zero 2
- **Focus**: IMU data classification, wireless control
- **Technologies**: Motion sensing, ML classification, MQTT

### 2. Voice-Activated Home Automation
Train keyword-spotting model on Pi, deploy to M5Core2 for local voice commands
- **Focus**: Audio processing, TinyML deployment
- **Technologies**: Audio features, CNN/MLP, edge inference

### 3. Low-Power Environmental Monitor
Stream sensor data to Pi for real-time analytics and dashboard visualization
- **Focus**: Streaming data, online learning
- **Technologies**: MQTT, data visualization, time-series analysis

### 4. Edge-Cloud Vision Classifier
Capture images on Pi camera, classify locally or via cloud, display on M5Core2
- **Focus**: CNNs, edge-cloud integration
- **Technologies**: Computer vision, distributed inference

### 5. Smart Anomaly Detection Node
Detect unusual motion/vibration patterns using IMU and unsupervised learning
- **Focus**: Anomaly detection, streaming analytics
- **Technologies**: Autoencoders, real-time processing

## 📝 Lab Documentation Requirements

All lab activities must be documented in GitHub Wiki including:
- Complete implementation process
- Screenshots and diagrams
- Demonstration videos
- Code snippets and explanations

## 📦 Final Deliverables

Each team must publish:
- ✅ **GitHub Repository** - complete code, data, and firmware
- ✅ **GitHub Wiki** - comprehensive project documentation
- ✅ **Video Demonstration** - working prototype showcase
- ✅ **Final Report (PDF)** - metrics, analysis, and visuals

## 🌟 Key Learning Highlights

- **Dual-Platform Approach**: Experience constraints of MCU (M5Core2) vs SBC (Pi Zero 2)
- **End-to-End Pipeline**: Collect on M5Core2 → Train on Pi → Deploy back
- **Cross-Device Networking**: Pi as edge gateway, M5Core2 as low-power node
- **Multi-Modal ML**: Vision on Pi, audio & IMU classification on M5Core2

## 🤝 Contributing

Please follow these guidelines for all lab submissions:
1. Create feature branches for each lab
2. Document thoroughly in Wiki
3. Include demo videos
4. Submit pull requests for review

## 📧 Contact

For questions about the course, please contact the instructor or teaching assistants.

---

**Course Duration**: February 2026 (15 days intensive)  
**Institution**: University of Nevada, Las Vegas (UNLV)  
**Program**: Embedded AI and Machine Learning
