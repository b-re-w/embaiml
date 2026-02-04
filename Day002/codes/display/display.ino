#include <M5Unified.h>

void setup() {
    // Initialize M5Stack with default settings
    auto cfg = M5.config();
    M5.begin(cfg);

    // Clear the screen to black
    M5.Display.fillScreen(BLACK);

    // 1. Korea: White, Medium Size
    M5.Display.setTextColor(WHITE);
    M5.Display.setTextSize(2); // Standard scaling
    M5.Display.setCursor(10, 30);
    M5.Display.print("Korea");

    // 2. China: Red, Larger Size
    // Using a different font style (FreeSans)
    M5.Display.setTextColor(RED);
    M5.Display.setFont(&fonts::FreeSansBold12pt7b); 
    M5.Display.setCursor(10, 80);
    M5.Display.print("China");

    // 3. USA: Blue, Custom Scale
    M5.Display.setTextColor(BLUE);
    M5.Display.setTextSize(4); // Very large
    M5.Display.setCursor(10, 150);
    M5.Display.print("USA");
}

void loop() {
    // Keep M5Unified internal state updated
    M5.update();
}