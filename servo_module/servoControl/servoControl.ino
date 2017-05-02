/*
Used arduino as the servo controllers because raspberry pi will
introduce jitter on the servo since it is not a real time kernel

Also, it is faster to delegate all the workload to the arduino than 
have one program always runnin on the pi
*/

#include <Servo.h>

Servo servoH;
Servo servoV;
int initialPos = 90;
int newPos = 90;
int angle = 0;
int baudRate = 9600;
int servoHPin = 11;
int servoVPin = 12;
long lastMillis = 0;
void setup() {
  
  Serial.begin(baudRate);  
  servoV.attach(servoVPin);
  servoV.write(angle);
  servoH.attach(servoHPin);
  servoH.write(angle);
  
}

void loop() {
  
  servoForward();
  servoBackward();
}

void servoBackward() {

  for(angle = 180; angle >= 1; angle -= 1) {
    Serial.print("180 - 0 ");
    Serial.println(angle);
    servoH.write(angle);
    servoV.write(angle);    
  }
}

void servoForward() {

  for(angle = 0; angle < 180; angle += 1) {

    Serial.print("0 - 180 ");
    Serial.println(angle);
    servoH.write(angle);  
    servoV.write(angle);     
  }
}

