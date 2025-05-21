#include <Wire.h>
#include <Adafruit_PWMServoDriver.h>

// Servo channels
#define PAN_CHANNEL  0
#define TILT_CHANNEL 4

// Pulse width bounds on PCA9685
#define SERVOMIN  150     
#define SERVOMAX  600     

Adafruit_PWMServoDriver pwm = Adafruit_PWMServoDriver();

void setup() {
  Serial.begin(115200);
  Wire.begin();
  pwm.begin();
  pwm.setPWMFreq(50);  
  // center servos on startup
  setPanAngle(120);
  setTiltAngle(10);
}

int angleToPulse(int angle, int angleMax) {
  // Map angle (0–angleMax) to pulse width
  return map(angle, 0, angleMax, SERVOMIN, SERVOMAX);
}

void setPanAngle(int angle) {
  angle = constrain(angle, 0, 180);
  int pulse = angleToPulse(angle, 180);
  pwm.setPWM(PAN_CHANNEL, 0, pulse);
}

void setTiltAngle(int angle) {
  angle = constrain(angle, 0, 180);
  int pulse = angleToPulse(angle, 180);
  pwm.setPWM(TILT_CHANNEL, 0, pulse);
}

void loop() {
    // Read from serial input
  if (Serial.available()) {
    String s = Serial.readStringUntil('\n');
    s.trim();
    int comma = s.indexOf(',');
    if (comma > 0) {
      int pan  = s.substring(0, comma).toInt();
      int tilt = s.substring(comma+1).toInt();
      setPanAngle(pan);
      setTiltAngle(tilt);
    }
  }
}
