PYRO - Vision Fire Detection  Program

1.Overview

	This program uses a deep learning model to analyze video streams from webcams or video files to detect fire presence
	and intensity. When fire is detected, the system activates visual and audio alerts based on the severity level.

2. System Requirements
	2.1 Hardware
		Operating System: Windows, Linux, or macOS (Windows recommended)
		Camera: Webcam for live detection
		Video Files: MP4 or AVI format support

	2.1 Software Dependencies
		Python 3.10
		TensorFlow/Keras framework
		Required Python libraries:
  			opencv-python` (video processing)
  			 `numpy` (numerical operations)
  			`tensorflow` (AI detection model)
  			 Audio library (see options below)

3. Quick Installation

	Install all core dependencies with a single command in PyCharm terminal (not Windows):
		"pip install opencv-python tensorflow numpy pygame"

4. Audio Library Options
	Choose one of the following audio libraries:
		Option 1: Windows (Recommended)
			`winsound` (built into Python on Windows)
		Option 2: Cross-platform
			bash
			pip install pygame

		Option 3: Alternative Cross-platform**
			bash  

			pip install playsound
5. Required Files
	Model File
		Filename: `fire_model.h5`
		Type: Pre-trained TensorFlow/Keras model
		Model Type : Binary classification (Fire/No Fire)
		Input Shape : (224, 224, 3) RGB images
		Output : Single probability value (0-1)

6. Usage Instructions
	6.1. Setup
   		 Download the required files: `fire_model.h5` and `main.py`
 		 Place both files in the same working directory
  		 Ensure the trained model file `fire_model.h5` is in the same folder as the script

	6.2. Run the Program
 		  Execute the main script to start the application

	6.3. Select Input Source
  		 Choose between webcam or video file input:
    			 - Webcam: Activates your camera for live fire detection
    			 - Video File: Upload a video from your device for analysis

	6.4. Monitor Detection
  		  A window displays the video feed with:
   			  - Detection bounding boxes
   			  - Fire level indicators
   			  - Confidence percentage bar

	6.5. Exit
   		- Press 'q' to quit the program

7. Contributing
	This is an academic project. For suggestions or collaboration inquiries, please contact me .

8. Contact
	For questions about this project:
		gmail : ayoubcherfaoui19.25@gmail.com
		Linkedin : https://www.linkedin.com/in/ayoub-cherfaoui-0b3699352/
