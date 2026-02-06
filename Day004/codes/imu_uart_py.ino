#include <M5Unified.h>

void setup() {
    auto cfg = M5.config();
    M5.begin(cfg);
    
    // Initialize Serial at 115200
    Serial.begin(115200);
    
    // Initialize IMU
    M5.Imu.init();
    
    M5.Display.setTextSize(2);
    M5.Display.clear();
    M5.Display.setCursor(0, 0);
    M5.Display.println("IMU Stream Active");
    M5.Display.println("100Hz @ 115200bps");
}

void loop() {
    static uint32_t next_ms = 0;
    uint32_t now = millis();

    // Trigger every 10ms (100Hz)
    if (now >= next_ms) {
        next_ms = now + 10;

        M5.Imu.update(); // Update the internal IMU state

        // In M5Unified, we fetch the data into specific float variables
        float ax, ay, az;
        float gx, gy, gz;

        M5.Imu.getAccel(&ax, &ay, &az);
        M5.Imu.getGyro(&gx, &gy, &gz);

        // Format: ts, ax, ay, az, gx, gy, gz
        // Using %.4f for higher precision often needed for HAR
        Serial.printf("%lu,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f\n", 
                      now, ax, ay, az, gx, gy, gz);
    }
}