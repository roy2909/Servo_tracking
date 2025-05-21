# PTZ Boat Tracking system
A Pan Tilt Zoom (PTZ) boat tracking system that uses YOLO for boat detection, servo based tracking via PID control, and digital zoom for closer observation. The system is designed to work with an Arduino, a pan tilt mount kit ith two servos, and a camera module. The system is capable of tracking boats in real time and providing a live video feed to the user. 


https://github.com/user-attachments/assets/3fb4d5b0-2746-4ac8-b335-ec20a7884e06


# Hardware Requirements
- Arduino board (e.g., Arduino Uno, Mega, etc.)
- Pan tilt mount kit with two servos
- Usb camera module 

# Dependencies
- Arduino IDE with Adafruit libraries
- OpenCV
- NumPy
- YOLOv8(Ultralytics)
  

# Usage
1. Upload the Arduino code to the Arduino board using the Arduino IDE.
2. Modify the configuration file to set the camera path, custom model path and other parameters.
3. Run the Python script `pyhton3 boat_tracking_servo.py` to start the boat tracking system.

