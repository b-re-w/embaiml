#include <M5Unified.h>
#include <MadgwickAHRS.h>

Madgwick filter;
unsigned long lastUpdate = 0;

void setup() {
    auto cfg = M5.config();
    M5.begin(cfg);
    Serial.begin(115200);

    // Madgwick filter needs a sample frequency (Hz)
    // We will target 50Hz
    filter.begin(50); 

    M5.Display.fillScreen(BLACK);
    M5.Display.setTextDatum(middle_center);
    M5.Display.setTextColor(MAGENTA);
    M5.Display.drawString("AHRS Full Euler (P/R/Y)", 160, 120);
}

void loop() {
    M5.update();

    float ax, ay, az;
    float gx, gy, gz;

    // 1. Get raw data
    M5.Imu.getAccel(&ax, &ay, &az);
    M5.Imu.getGyro(&gx, &gy, &gz);

    // 2. Update the filter
    // Note: Madgwick usually expects gyro in degrees/sec and accel in Gs
    filter.updateIMU(gx, gy, gz, ax, ay, az);

    // 3. Get the calculated angles
    float roll  = filter.getRoll();
    float pitch = filter.getPitch();
    float yaw   = filter.getYaw();

    // 4. Output as CSV to UART
    Serial.printf("%.2f,%.2f,%.2f\n", pitch, roll, yaw);

    // Maintain a consistent 50Hz update rate
    while (millis() - lastUpdate < 20) {
        delay(1);
    }
    lastUpdate = millis();
}