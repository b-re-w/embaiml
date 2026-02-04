#include <M5Unified.h>
#include <SD.h>

// Audio Settings
#define SAMPLE_RATE 16000
#define CHANNELS 1
#define BITS_PER_SAMPLE 16

File recFile;
bool isRecording = false;
uint32_t dataSize = 0;
const char* filename = "/recording.wav";

// Standard WAV Header (44 bytes)
struct wav_header_t {
    char riff[4] = {'R', 'I', 'F', 'F'};
    uint32_t fileSize;
    char wave[4] = {'W', 'A', 'V', 'E'};
    char fmt[4] = {'f', 'm', 't', ' '};
    uint32_t fmtSize = 16;
    uint16_t audioFormat = 1; // PCM
    uint16_t numChannels = CHANNELS;
    uint32_t sampleRate = SAMPLE_RATE;
    uint32_t byteRate = SAMPLE_RATE * CHANNELS * BITS_PER_SAMPLE / 8;
    uint16_t blockAlign = CHANNELS * BITS_PER_SAMPLE / 8;
    uint16_t bitsPerSample = BITS_PER_SAMPLE;
    char data[4] = {'d', 'a', 't', 'a'};
    uint32_t dataSize;
};

void writeHeader() {
    wav_header_t header;
    header.fileSize = dataSize + 36;
    header.dataSize = dataSize;
    recFile.seek(0);
    recFile.write((uint8_t*)&header, sizeof(wav_header_t));
}

void setup() {
    auto cfg = M5.config();
    M5.begin(cfg);

    // Initialize SD Card
    if (!SD.begin(GPIO_NUM_4, SPI, 40000000)) {
        M5.Display.println("SD Error!");
        while (1);
    }

    // Configure Microphone
    auto mic_cfg = M5.Mic.config();
    mic_cfg.sample_rate = SAMPLE_RATE;
    M5.Mic.config(mic_cfg);
    M5.Mic.begin();

    M5.Display.setTextDatum(middle_center);
    M5.Display.drawString("Button A: Record", 160, 120);
}

void loop() {
    M5.update();

    if (M5.BtnA.wasPressed()) {
        isRecording = !isRecording;
        if (isRecording) {
            M5.Display.fillScreen(RED);
            M5.Display.drawString("RECORDING...", 160, 120);
            
            dataSize = 0;
            recFile = SD.open(filename, FILE_WRITE);
            // Write temporary blank header
            recFile.seek(sizeof(wav_header_t)); 
        } else {
            M5.Display.fillScreen(BLACK);
            M5.Display.drawString("SAVING...", 160, 120);
            
            writeHeader(); // Finalize file size in header
            recFile.close();
            
            M5.Display.fillScreen(BLACK);
            M5.Display.drawString("Done! A to Rec again", 160, 120);
        }
    }

    if (isRecording) {
        int16_t buffer[512];
        // Read from Mic and write to SD
        if (M5.Mic.record(buffer, 512, SAMPLE_RATE)) {
            recFile.write((uint8_t*)buffer, 512 * sizeof(int16_t));
            dataSize += 512 * sizeof(int16_t);
        }
    }
}