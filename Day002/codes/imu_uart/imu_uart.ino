#include <M5Unified.h>

void setup() {
    auto cfg = M5.config();
    M5.begin(cfg);

    // Initialize Serial for UART communication (default baud rate is 115200)
    // No need for Serial.begin() as M5.begin() handles it, but you can set it:
    Serial.begin(115200);

    M5.Display.fillScreen(BLACK);
    M5.Display.setTextDatum(middle_center);
    M5.Display.setTextColor(GREEN);
    M5.Display.drawString("Streaming IMU to UART...", 160, 120);
    M5.Display.drawString("Baud: 115200", 160, 150);
}

void loop() {
    M5.update();

    // Variables for the 6 data points
    float ax, ay, az; // Accelerometer (Gs)
    float gx, gy, gz; // Gyroscope (deg/s)

    // Fetch data from the IMU
    M5.Imu.getAccel(&ax, &ay, &az);
    M5.Imu.getGyro(&gx, &gy, &gz);

    // Print to UART as CSV: ax,ay,az,gx,gy,gz
    // %.4f provides 4 decimal places for high precision
    Serial.printf("%.4f,%.4f,%.4f,%.4f,%.4f,%.4f\n", ax, ay, az, gx, gy, gz);

    // Short delay to prevent flooding the Serial buffer too fast
    // 20ms = 50Hz refresh rate
    delay(20); 
}