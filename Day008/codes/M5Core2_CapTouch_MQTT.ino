#include <M5Unified.h>       // Optimized for Core2
#include <WiFi.h>
#include <PubSubClient.h>
#include <ArduinoJson.h>     // For structured JSON payloads

// --- USER CONFIGURATION ---
const char* ssid = "YOUR_WIFI_SSID";
const char* password = "YOUR_WIFI_PASSWORD";
const char* mqtt_server = "192.168.1.100"; // RPi Zero 2 W IP
const int mqtt_port = 1883;
const char* location = "Lab_M5_Unit1";
// --------------------------

WiFiClient espClient;
PubSubClient client(espClient);

void setup_wifi() {
  M5.Display.print("Connecting to WiFi...");
  WiFi.begin(ssid, password);
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    M5.Display.print(".");
  }
  M5.Display.println("\nWiFi connected");
}

void reconnect() {
  while (!client.connected()) {
    M5.Display.print("Attempting MQTT connection...");
    String clientId = "M5Core2-";
    clientId += String((uint32_t)ESP.getEfuseMac(), HEX);

    if (client.connect(clientId.c_str())) {
      M5.Display.println("connected");
    } else {
      M5.Display.printf("failed, rc=%d try again in 5s\n", client.state());
      delay(5000);
    }
  }
}

void setup() {
  auto cfg = M5.config();
  M5.begin(cfg);
  M5.Display.setTextSize(2);

  setup_wifi();
  client.setServer(mqtt_server, mqtt_port);

  M5.Display.fillScreen(TFT_BLACK);
  M5.Display.setCursor(0, 50);
  M5.Display.println("Firm press = High Val");
  M5.Display.println("Light tap = Low Val");
}

void loop() {
  M5.update();

  if (!client.connected()) {
    reconnect();
  }
  client.loop();

  // Get touch details
  auto detail = M5.Touch.getDetail();

  // Trigger only when initial touch is detected
  if (detail.wasPressed()) {
    // We use the 'size' of the touch area as a proxy for pressure
    int touchSize = detail.size;

    // 1. Generate Random Data modulated by Touch Size
    // Higher touchSize results in higher offset for temp and hum
    float t_val = 20.0 + (rand() % 50) / 10.0 + (touchSize / 10.0);
    float h_val = 40.0 + (rand() % 100) / 10.0 + (touchSize / 5.0);

    // 2. Create JSON Payload
    StaticJsonDocument<256> doc;
    doc["loc"] = location;
    doc["temp"] = serialized(String(t_val, 2));
    doc["hum"] = serialized(String(h_val, 2));
    doc["raw_size"] = touchSize; // Log the area size used for modulation

    char buffer[256];
    serializeJson(doc, buffer);

    // 3. Publish to Raspberry Pi
    if(client.publish("lab/temphum", buffer)) {
        // 4. Visual Feedback
        M5.Display.fillScreen(TFT_DARKGREEN);
        M5.Display.setCursor(10, 50);
        M5.Display.printf("Size Detected: %d\n", touchSize);
        M5.Display.printf("Temp: %.2f C\n", t_val);
        M5.Display.printf("Hum:  %.2f %%", h_val);

        delay(1000); // UI Pause
        M5.Display.fillScreen(TFT_BLACK);
        M5.Display.setCursor(0, 50);
        M5.Display.println("Touch screen to send");
    }
  }
}