# Mothusi: AI-Powered Assistive Smart Assistant

**Mothusi** is an AI-powered IoT system designed to assist individuals with disabilities by providing real-time object recognition and voice command feedback. This project aims to enhance accessibility and independence for users with visual impairments or motor difficulties by integrating AI, embedded systems, and IoT technologies.

## Introduction

With advancements in AI, embedded systems, and IoT, Mothusi leverages these technologies to help people interact more easily with their surroundings. The system uses a camera for object detection, a microphone for voice command recognition, and a speaker to provide audio feedback, making it responsive and practical for everyday use.

## Project Overview

Mothusi fulfills three primary functions:
1. **Object Recognition**: Detects objects using a pre-trained AI model like YOLO, MobileNet, or TensorFlow Lite, with voice feedback for identified objects.
2. **Voice Command Recognition**: Accepts basic voice commands (e.g., "What is this?", "Identify object") and processes them using pre-trained models or APIs like TensorFlow Lite or Google Speech API.
3. **Voice Feedback**: Provides real-time audio descriptions of detected objects, assisting users in identifying their surroundings.

## Milestones and Deliverables

### Milestone 1: Initial Analysis, Design, and Basic Object Recognition
**Due Date**: October 14, 2024

**Objectives**:
- Analyze system requirements for object recognition, voice command processing, and feedback.
- Design initial block diagrams/flowcharts showing interactions between the camera, microphone, speaker, and embedded system.
- Implement basic object recognition using a pre-trained model to detect objects from a camera feed.

**Deliverables**:
- **Analysis Document**: Overview of system requirements.
- **Initial Design**: Interaction diagrams.
- **Core Feature 1**: Basic object recognition (e.g., using YOLO).
- **Updated Repository**: Code and initial documentation.

### Milestone 2: Voice Command Recognition and System Refinement
**Due Date**: October 28, 2024

**Objectives**:
- Refine the initial design based on feedback.
- Add voice command recognition for commands like "Identify object."
- Ensure integration of object and voice processing, allowing voice commands to initiate object recognition.

**Deliverables**:
- **Refined Design**: Updated design documents.
- **Core Feature 2**: Basic speech recognition for commands.
- **Integration of Object and Voice Processing**.
- **Updated Repository**: Documentation and instructions for running the combined system.

### Milestone 3: Voice Feedback and Final Integration
**Due Date**: November 18, 2024

**Objectives**:
- Finalize system integration for smooth operation across object recognition, voice command processing, and voice feedback.
- Implement voice feedback to announce recognized objects through a speaker.
- Conduct testing and optimize for real-time performance.

**Deliverables**:
- **Final System Integration**: Fully integrated functionality.
- **Core Feature 3**: Voice feedback.
- **Testing and Optimizations**.
- **Final Repository**: Complete with source code, README, and contact information.

## Setup and Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/KabeloLebitsa/Mothusi-AI.git
   cd Mothusi-AI
   ```

2. **Install dependencies**:
   ```bash
   pip install ultralytics opencv-python numpy matplotlib tensorflow tensorflow-lite
   ```

3. **Download pre-trained models**: Ensure models like `yolov8m.pt` and required audio models are in the project directory.

## Usage

To start Mothusi:
1. Run the main script:
   ```bash
   python3 mothusi.py
   ```
2. Interact with Mothusi using voice commands or by placing objects in front of the camera.

### Example Commands
- **"What is this?"**: Triggers object recognition.
- **"Identify object"**: Initiates identification.

## Contributors

- **Kabelo Lebitsa**
- **Tsoanelo Rameno**

## License

This project is licensed under the MIT License.
