#include <M5Unified.h>

int count = 0;

void setup() {
    auto cfg = M5.config();
    M5.begin(cfg);

    M5.Display.fillScreen(BLACK);
    M5.Display.setTextDatum(middle_center);
    
    updateDisplay();
}

void loop() {
    M5.update(); 

    // 1. Short Press: Increment
    if (M5.BtnPWR.wasClicked()) {
        count++;
        updateDisplay();
    }

    // 2. Long Press Logic:
    // wasReleaseFor(ms) returns true if the button was held for AT LEAST ms.
    if (M5.BtnPWR.wasReleaseFor(2000)) {
        // Now we check if it was released BEFORE 5000ms.
        // In M5Unified, we can check the current duration since the last change.
        if (M5.BtnPWR.lastChange() < 5000) {
            count = 0;
            updateDisplay();
        }
    }
}

void updateDisplay() {
    M5.Display.fillScreen(BLACK);
    M5.Display.setTextColor(WHITE, BLACK);
    
    M5.Display.setFont(&fonts::FreeSansBold24pt7b);
    M5.Display.drawNumber(count, 160, 110);
    
    M5.Display.setFont(&fonts::FreeSans9pt7b);
    M5.Display.setTextColor(ORANGE, BLACK);
    M5.Display.drawString("Click: +1 | Hold 2-5s: Reset", 160, 200);
}