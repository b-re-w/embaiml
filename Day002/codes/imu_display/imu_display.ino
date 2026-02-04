#include <M5Unified.h>

void setup() {
    auto cfg = M5.config();
    M5.begin(cfg);

    M5.Display.fillScreen(BLACK);
    M5.Display.setTextDatum(top_left);
    M5.Display.setFont(&fonts::FreeSansBold12pt7b);
    M5.Display.setTextColor(WHITE, BLACK);
}

void loop() {
    M5.update();

    // Variables to hold the sensor readings
    float ax, ay, az;
    float gx, gy, gz;

    // Fetch Accelerometer and Gyroscope data
    // These functions update the float variables directly
    M5.Imu.getAccel(&ax, &ay, &az);
    M5.Imu.getGyro(&gx, &gy, &gz);

    // Reset cursor to overwrite the previous frame
    M5.Display.setCursor(0, 0);
    
    // 1. Display Accelerometer Data (Gs)
    M5.Display.setTextColor(YELLOW, BLACK);
    M5.Display.println("--- ACCEL (G) ---");
    M5.Display.printf("X: %6.2f  \n", ax);
    M5.Display.printf("Y: %6.2f  \n", ay);
    M5.Display.printf("Z: %6.2f  \n\n", az);

    // 2. Display Gyroscope Data (deg/s)
    M5.Display.setTextColor(CYAN, BLACK);
    M5.Display.println("--- GYRO (deg/s) ---");
    M5.Display.printf("X: %6.2f  \n", gx);
    M5.Display.printf("Y: %6.2f  \n", gy);
    M5.Display.printf("Z: %6.2f  \n", gz);

    delay(50); // High refresh rate for smooth monitoring
}