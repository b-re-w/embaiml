#include <M5Unified.h>

void setup() {
    auto cfg = M5.config();
    M5.begin(cfg);

    // Initial Screen Setup
    M5.Display.fillScreen(BLACK);
    M5.Display.setTextDatum(middle_center);
    M5.Display.setFont(&fonts::FreeSansBold12pt7b);
    M5.Display.drawString("Touch a button below", 160, 120);
}

void loop() {
    // Required to refresh touch sensor data
    M5.update();

    // Check Button A (Left Circle)
    if (M5.BtnA.wasPressed()) {
        displayMessage("BUTTON A", RED);
    }

    // Check Button B (Middle Circle)
    if (M5.BtnB.wasPressed()) {
        displayMessage("BUTTON B", GREEN);
    }

    // Check Button C (Right Circle)
    if (M5.BtnC.wasPressed()) {
        displayMessage("BUTTON C", BLUE);
    }
}

void displayMessage(const char* msg, uint16_t color) {
    M5.Display.fillScreen(BLACK); // Clear screen before next display
    M5.Display.setTextColor(color);
    M5.Display.setFont(&fonts::FreeSansBold18pt7b);
    M5.Display.drawString(msg, 160, 120);
}