#include <M5Unified.h>
#include <SD.h>
#include <MadgwickAHRS.h>
#include <math.h>

Madgwick filter;
bool isRecording = false;
File logFile;
const char* filename = "/imu_quat.csv";
unsigned long lastUpdate = 0;

void setup() {
    auto cfg = M5.config();
    M5.begin(cfg);

    if (!SD.begin(GPIO_NUM_4, SPI, 40000000)) {
        M5.Display.println("SD Card Error!");
        while (1); 
    }

    filter.begin(50); 
    M5.Display.setTextDatum(middle_center);
    M5.Display.setFont(&fonts::FreeSansBold12pt7b);
    updateUI();
}

void loop() {
    M5.update();

    if (M5.BtnA.wasPressed()) {
        isRecording = !isRecording;
        if (isRecording) {
            if (!SD.exists(filename)) {
                logFile = SD.open(filename, FILE_WRITE);
                logFile.println("ms,ax,ay,az,gx,gy,gz,qw,qx,qy,qz");
                logFile.close();
            }
        }
        updateUI();
    }

    if (millis() - lastUpdate >= 20) {
        lastUpdate = millis();
        processIMU();
    }
}

void processIMU() {
    float ax, ay, az, gx, gy, gz;
    M5.Imu.getAccel(&ax, &ay, &az);
    M5.Imu.getGyro(&gx, &gy, &gz);

    filter.updateIMU(gx, gy, gz, ax, ay, az);

    if (isRecording) {
        // Step 1: Get Euler Angles (supported by all Madgwick versions)
        // Values are in degrees, so we convert to radians for math
        float roll  = filter.getRoll()  * M_PI / 180.0f;
        float pitch = filter.getPitch() * M_PI / 180.0f;
        float yaw   = filter.getYaw()   * M_PI / 180.0f;

        // Step 2: Convert Euler to Quaternion
        float cy = cos(yaw * 0.5);
        float sy = sin(yaw * 0.5);
        float cp = cos(pitch * 0.5);
        float sp = sin(pitch * 0.5);
        float cr = cos(roll * 0.5);
        float sr = sin(roll * 0.5);

        float qw = cr * cp * cy + sr * sp * sy;
        float qx = sr * cp * cy - cr * sp * sy;
        float qy = cr * sp * cy + sr * cp * sy;
        float qz = cr * cp * sy - sr * sp * cy;

        logFile = SD.open(filename, FILE_APPEND);
        if (logFile) {
            logFile.printf("%lu,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.4f,%.4f,%.4f,%.4f\n", 
                            millis(), ax, ay, az, gx, gy, gz, qw, qx, qy, qz);
            logFile.close();
        }
    }
}

void updateUI() {
    M5.Display.fillScreen(isRecording ? RED : BLACK);
    M5.Display.setTextColor(WHITE);
    M5.Display.drawString(isRecording ? "LOGGING QUAT" : "IDLE", 160, 100);
    M5.Display.drawString("Button A to Toggle", 160, 140);
}