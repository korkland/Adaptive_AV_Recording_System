# Adaptive Automotive Recording System

An intelligent edge AI system for automotive environments that uses Vision Transformers to detect configurable events and perform retroactive recording with remote configuration management.

## Project Overview

This system combines real-time computer vision with adaptive event detection, allowing remote configuration of scene queries while maintaining efficient edge processing. When specified events are detected, the system automatically saves retroactive footage for configurable time periods.

### Key Features
- **Vision Transformer Integration**: Utilizes PaliGemma VIT architectures for scene understanding
- **Dynamic Query Configuration**: Receive and update event detection queries via OTA updates
- **Retroactive Recording**: Buffer management with configurable lookback periods
- **Edge Optimization**: PyTorch to ONNX to TensorRT deployment pipeline
- **Remote Management**: Back-office integration for configuration updates

## System Architecture

```
┌──────────────────┐     ┌───────────────────┐     ┌──────────────────┐
│   Camera Feed    │     │   Buffer Manager  │     │   Back Office    │
│                  │ ──▶ │                   │ ◀── │                  │
│  (Live Stream)   │     │ (Circular Buffer) │     │ (Config/Queries) │
└──────────────────┘     └───────────────────┘     └──────────────────┘
         │                         │                        │
         ▼                         ▼                        ▼
┌──────────────────┐     ┌───────────────────┐     ┌──────────────────┐
│    VIT Model     │     │ Event Processor   │     │    OTA Manager   │
│                  │ ──▶ │                   │ ◀── │                  │
│ (Scene Analysis) │     │ (Query Matching)  │     │                  │
└──────────────────┘     └───────────────────┘     └──────────────────┘
         │                         │
         ▼                         ▼
┌──────────────────┐     ┌───────────────────┐
│   ONNX Runtime   │     │  Recording Engine │
│                  │     │                   │
│   (Inference)    │     │   (Retroactive)   │
└──────────────────┘     └───────────────────┘
```

## Development Roadmap

### Phase 1: Core Infrastructure (Weeks 1-3)
- [ ] **Environment Setup**
  - Set up PyTorch development environment
  - Install ONNX and TensorRT dependencies
  - Configure automotive hardware testing environment

- [ ] **Basic Video Pipeline**
  - Implement camera interface (V4L2/GStreamer)
  - Create circular buffer for video frames
  - Basic frame preprocessing and normalization

- [ ] **Model Integration**
  - Research and select VIT architecture (ViT-Base, DeiT, or custom)
  - Implement PyTorch model loading and inference
  - Create basic scene analysis pipeline

### Phase 2: Event Detection System (Weeks 4-6)
- [ ] **Query Engine Development**
  - Design query language/format for event descriptions
  - Implement natural language to vision query mapping
  - Create event matching algorithms

- [ ] **Configuration Management**
  - Design configuration schema (JSON/YAML)
  - Implement configuration parsing and validation
  - Create query update mechanisms

- [ ] **Event Processing**
  - Develop confidence scoring for detections
  - Implement threshold-based triggering
  - Create event logging and metadata capture

### Phase 3: Recording & Storage (Weeks 7-8)
- [ ] **Retroactive Recording**
  - Implement configurable lookback periods
  - Develop efficient video segment extraction
  - Create metadata association with recordings

- [ ] **Storage Management**
  - Design storage hierarchy (local/cloud)
  - Implement compression and encoding
  - Create cleanup and retention policies

### Phase 4: Remote Configuration (Weeks 9-10)
- [ ] **OTA System**
  - Design secure communication protocol
  - Implement configuration download/update
  - Create rollback mechanisms for failed updates

- [ ] **Back Office Integration**
  - Design REST API for configuration management
  - Implement authentication and authorization
  - Create monitoring and health reporting

### Phase 5: Optimization & Deployment (Weeks 11-14)
- [ ] **Model Optimization**
  - Convert PyTorch models to ONNX format
  - Optimize ONNX models for TensorRT
  - Implement quantization and pruning

- [ ] **Performance Tuning**
  - Profile inference performance
  - Optimize memory usage and buffer management
  - Implement GPU acceleration where available

- [ ] **Production Deployment**
  - Create Docker containers for deployment
  - Implement systemd services for auto-start
  - Develop monitoring and logging systems

### Phase 6: Testing & Validation (Weeks 15-16)
- [ ] **Integration Testing**
  - End-to-end system testing
  - Performance benchmarking
  - Edge case handling validation

- [ ] **Field Testing**
  - Automotive environment testing
  - Real-world scenario validation
  - Performance optimization based on results

## Project Structure

```
adaptive-recording-system/
├── README.md
├── requirements.txt
├── setup.py
├── config/
│   ├── default_config.yaml
│   ├── model_configs/
│   └── deployment_configs/
├── src/
│   ├── __init__.py
│   ├── core/
│   │   ├── camera_manager.py
│   │   ├── buffer_manager.py
│   │   └── inference_engine.py
│   ├── models/
│   │   ├── vit_wrapper.py
│   │   ├── onnx_runtime.py
│   │   └── trt_inference.py
│   ├── detection/
│   │   ├── query_engine.py
│   │   ├── event_processor.py
│   │   └── scene_analyzer.py
│   ├── recording/
│   │   ├── retroactive_recorder.py
│   │   ├── storage_manager.py
│   │   └── metadata_handler.py
│   ├── communication/
│   │   ├── ota_manager.py
│   │   ├── config_updater.py
│   │   └── backoffice_client.py
│   └── utils/
│       ├── logging_config.py
│       ├── performance_monitor.py
│       └── system_health.py
├── tests/
│   ├── unit/
│   ├── integration/
│   └── performance/
├── deployment/
│   ├── docker/
│   ├── systemd/
│   └── kubernetes/
├── models/
│   ├── pytorch/
│   ├── onnx/
│   └── tensorrt/
├── docs/
│   ├── api_reference.md
│   ├── deployment_guide.md
│   └── configuration_guide.md
└── examples/
    ├── basic_usage.py
    ├── custom_queries.py
    └── deployment_examples/
```

## Technical Stack

### Core Components
- **Computer Vision**: PyTorch, torchvision, OpenCV
- **Model Deployment**: ONNX Runtime, TensorRT
- **Video Processing**: FFmpeg, GStreamer
- **Communication**: gRPC, HTTP/REST APIs
- **Configuration**: YAML/JSON, Pydantic for validation

### Development Tools
- **Languages**: Python 3.9+, C++ (for TensorRT optimization)
- **Testing**: pytest, unittest, performance profiling tools
- **Deployment**: Docker, Kubernetes (optional)
- **Monitoring**: Prometheus, Grafana (optional)

## Configuration Schema

```yaml
system:
  model_path: "models/vit_base.pth"
  inference_device: "cuda:0"
  buffer_size_minutes: 5
  recording_lookback_seconds: 30

queries:
  - id: "pedestrian_crossing"
    description: "Person crossing in front of vehicle"
    confidence_threshold: 0.8
    enabled: true
  - id: "aggressive_driving"
    description: "Rapid lane changes or tailgating"
    confidence_threshold: 0.7
    enabled: true

recording:
  video_quality: "1080p"
  compression: "h264"
  max_storage_gb: 100
  retention_days: 30

communication:
  backoffice_url: "https://api.fleet-management.com"
  update_interval_minutes: 15
  heartbeat_interval_minutes: 5
```

## Quick Start

### Installation
```bash
# Clone repository
git clone https://github.com/korkland/adaptive-recording-system.git
cd adaptive-recording-system

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install in development mode
pip install -e .
```

### Basic Usage
```python
from src.core.inference_engine import AdaptiveRecordingSystem

# Initialize system
system = AdaptiveRecordingSystem(config_path="config/default_config.yaml")

# Start processing
system.start()

# System will now:
# 1. Process camera feed
# 2. Run VIT inference
# 3. Check for configured events
# 4. Record retroactively when events detected
# 5. Listen for configuration updates
```

## Requirements

### Hardware Requirements
- **GPU**: NVIDIA GPU with CUDA support (RTX 3060+ recommended)
- **RAM**: 8GB minimum, 16GB recommended
- **Storage**: 500GB+ for video buffer and recordings
- **Camera**: USB 3.0 or automotive camera interface

### Software Requirements
- **OS**: Ubuntu 20.04+ (primary), Windows 10+ (development)
- **Python**: 3.9+
- **CUDA**: 11.8+
- **TensorRT**: 8.5+

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.