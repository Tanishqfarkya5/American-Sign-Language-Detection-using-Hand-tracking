# **American Sign Language (ASL) Recognition System**

## **Project Overview**
The **ASL Recognition System** is a real-time application designed to detect and classify American Sign Language (ASL) alphabet signs (A-Z). It uses advanced deep learning techniques alongside computer vision and hand tracking to provide accurate and robust predictions. This project is aimed at promoting inclusivity and improving communication for individuals with hearing or speech impairments.

## **Features**
- **Deep Learning Model**:
  - A custom Convolutional Neural Network (CNN) designed for ASL alphabet classification.
  - Achieved high accuracy through data augmentation and advanced training techniques.
  
- **Hand Tracking**:
  - Integrated **Mediapipe** for efficient hand landmark detection and preprocessing.
  - Dynamically extracts regions of interest (ROI) from live webcam feeds for recognition.

- **Real-Time Recognition**:
  - Utilized **OpenCV** to capture and process live video streams.
  - Provides real-time feedback on predicted ASL alphabets.

- **User-Friendly GUI**:
  - Built with **Tkinter** for easy navigation, allowing users to train and test the model seamlessly.

## **Technologies Used**
- **Programming Language**: Python
- **Libraries**: TensorFlow, Keras, NumPy, OpenCV, Mediapipe, Tkinter
- **Machine Learning**: Convolutional Neural Networks (CNNs)

## **Setup and Installation**
### **Prerequisites**:
Ensure the following libraries are installed:
- TensorFlow
- Keras
- OpenCV
- Mediapipe
- NumPy
- Tkinter (pre-installed with Python)

## **Usage**
### **Train the Model**
- Load the training dataset (ASL alphabet images).
- Start the training process using the built-in GUI.

### **Test the Model**
- Use a live webcam feed for real-time testing.
- Recognized letters are displayed on the video feed.

### **Customize**
- Replace the dataset with custom ASL gestures or expand the model to include additional signs.

## **Dataset**
The project uses the **ASL Alphabet Dataset**, which contains images for 29 classes (A-Z and space).  
**Link to Dataset:** [ASL Alphabet](https://www.kaggle.com/grassknoted/asl-alphabet)

## **Future Enhancements**
- Extend the model to recognize ASL words or sentences.
- Integrate audio output for recognized signs to improve accessibility.
- Enhance recognition accuracy with additional training data and fine-tuned models.
- Develop a mobile application for wider usage.

## **Acknowledgments**
- Dataset sourced from [Kaggle](https://www.kaggle.com/grassknoted/asl-alphabet).
- Special thanks to the creators of TensorFlow, Mediapipe, and OpenCV for enabling this project.
