#include <M5Unified.h>
#include <MadgwickAHRS.h>

Madgwick filter;
unsigned long lastUpdate = 0;
const float sampleFreq = 50.0f; // 50Hz update rate

void setup() {
    auto cfg = M5.config();
    M5.begin(cfg);

    Serial.begin(115200);
    
    // Initialize the filter
    filter.begin(sampleFreq);

    M5.Display.fillScreen(BLACK);
    M5.Display.setTextDatum(middle_center);
    M5.Display.setTextColor(YELLOW);
    M5.Display.drawString("MADGWICK UART STREAM", 160, 120);
}

void loop() {
    M5.update();

    // Maintain a steady 50Hz timing for the filter math
    if (millis() - lastUpdate >= (1000 / sampleFreq)) {
        lastUpdate = millis();

        float ax, ay, az;
        float gx, gy, gz;
        
        // 1. Get Raw Data
        M5.Imu.getAccel(&ax, &ay, &az);
        M5.Imu.getGyro(&gx, &gy, &gz);

        // 2. Update Filter (Gyro: deg/s, Accel: Gs)
        filter.updateIMU(gx, gy, gz, ax, ay, az);

        // 3. Get Euler Angles
        float roll  = filter.getRoll();
        float pitch = filter.getPitch();
        float yaw   = filter.getYaw();

        // 4. Stream to Python via UART
        // Format: ms, ax, ay, az, gx, gy, gz, roll, pitch, yaw
        Serial.printf("%lu,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.2f,%.2f,%.2f\n", 
                      millis(), ax, ay, az, gx, gy, gz, roll, pitch, yaw);
    }
}