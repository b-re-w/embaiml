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
unsigned long lastMsg = 0;

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
    // Generate unique ID based on the device MAC address
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
}

void loop() {
  M5.update(); // Maintain button and power states

  if (!client.connected()) {
    reconnect();
  }
  client.loop();

  unsigned long now = millis();
  // Publish every 5 seconds without using blocking delay()
  if (now - lastMsg > 5000) {
    lastMsg = now;

    // 1. Generate Nominal Random Data
    float t_val = 20.0 + (rand() % 100) / 10.0; // Range: 20.0 - 30.0 C
    float h_val = 40.0 + (rand() % 200) / 10.0; // Range: 40.0 - 60.0 %

    // 2. Create JSON Payload
    StaticJsonDocument<256> doc;
    doc["loc"] = location;
    doc["temp"] = serialized(String(t_val, 2)); // 2 decimal places
    doc["hum"] = serialized(String(h_val, 2));

    char buffer[256];
    serializeJson(doc, buffer);

    // 3. Publish to the Raspberry Pi
    client.publish("lab/temphum", buffer);

    // 4. Visual Feedback on M5Core2 Screen
    M5.Display.fillScreen(TFT_BLACK);
    M5.Display.setCursor(10, 50);
    M5.Display.printf("Loc: %s\n", location);
    M5.Display.printf("Temp: %.2f C\n", t_val);
    M5.Display.printf("Hum:  %.2f %%", h_val);
  }
}