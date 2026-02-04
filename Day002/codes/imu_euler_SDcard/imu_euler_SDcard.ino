#include <M5Unified.h>
#include <SD.h>

bool isRecording = false;
File logFile;
const char* filename = "/imu_log.csv";

void setup() {
    auto cfg = M5.config();
    M5.begin(cfg);

    // Initialize SD Card
    // M5Core2 uses specific pins (G18, G23, G19, G4)
    if (!SD.begin(GPIO_NUM_4, SPI, 40000000)) {
        M5.Display.println("SD Card Mount Failed!");
        while (1); // Stop execution if no SD
    }

    M5.Display.setTextDatum(middle_center);
    M5.Display.setFont(&fonts::FreeSansBold12pt7b);
    updateUI();
}

void loop() {
    M5.update();

    // Toggle recording when Button A (left circle) is pressed
    if (M5.BtnA.wasPressed()) {
        isRecording = !isRecording;
        
        if (isRecording) {
            // Write Header if it's a new file
            if (!SD.exists(filename)) {
                logFile = SD.open(filename, FILE_WRITE);
                logFile.println("Timestamp_ms,ax,ay,az,gx,gy,gz,pitch,roll");
                logFile.close();
            }
        }
        updateUI();
    }

    if (isRecording) {
        saveData();
    }
}

void saveData() {
    float ax, ay, az, gx, gy, gz;
    M5.Imu.getAccel(&ax, &ay, &az);
    M5.Imu.getGyro(&gx, &gy, &gz);

    // Calculate Euler angles (Pitch and Roll)
    float pitch = atan2(-ax, sqrt(ay * ay + az * az)) * 180.0 / M_PI;
    float roll  = atan2(ay, az) * 180.0 / M_PI;

    // Open file for appending
    logFile = SD.open(filename, FILE_APPEND);
    if (logFile) {
        logFile.printf("%lu,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.2f,%.2f\n", 
                        millis(), ax, ay, az, gx, gy, gz, pitch, roll);
        logFile.close();
    }
}

void updateUI() {
    M5.Display.fillScreen(isRecording ? RED : BLACK);
    M5.Display.setTextColor(WHITE);
    if (isRecording) {
        M5.Display.drawString("RECORDING...", 160, 100);
        M5.Display.drawString("Press A to Stop", 160, 140);
    } else {
        M5.Display.drawString("IDLE", 160, 100);
        M5.Display.drawString("Press A to Record", 160, 140);
    }
}