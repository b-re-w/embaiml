#include <M5Unified.h>

void setup() {
    auto cfg = M5.config();
    M5.begin(cfg);

    M5.Display.fillScreen(BLACK);
    M5.Display.setTextColor(YELLOW);
    M5.Display.setFont(&fonts::FreeSans12pt7b);
    M5.Display.setCursor(10, 30);
    M5.Display.print("Random Number:");
}

void loop() {
    M5.update();

    float randomVal = random(1000000) / 1000.0;

    // Set cursor to the same spot every time
    M5.Display.setCursor(10, 100);
    M5.Display.setFont(&fonts::FreeSansBold18pt7b);

    // CRITICAL FIX: Set text color AND background color (BLACK)
    // This tells the display to "wipe" the area behind each character
    M5.Display.setTextColor(CYAN, BLACK); 

    // Print with 3 decimal precision
    M5.Display.printf("%.3f   ", randomVal); // Extra spaces clear trailing digits

    delay(1000); 
}