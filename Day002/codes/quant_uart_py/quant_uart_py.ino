#include <M5Unified.h>
#include <math.h>

// Quaternion state variables (Start at Identity: [1, 0, 0, 0])
float qw = 1.0f, qx = 0.0f, qy = 0.0f, qz = 0.0f;

// Timing
unsigned long lastUpdate = 0;
const float alpha = 0.98f; // Filter coefficient (Weight of Gyro vs Accel)

void setup() {
    auto cfg = M5.config();
    M5.begin(cfg);
    Serial.begin(115200);
}

void loop() {
    M5.update();
    
    unsigned long now = micros();
    float dt = (now - lastUpdate) / 1000000.0f; // Delta time in seconds
    lastUpdate = now;

    // 1. Get Raw Data
    float ax, ay, az, gx, gy, gz;
    M5.Imu.getAccel(&ax, &ay, &az);
    M5.Imu.getGyro(&gx, &gy, &gz);

    // Convert Gyro from deg/s to rad/s
    float gx_rad = gx * M_PI / 180.0f;
    float gy_rad = gy * M_PI / 180.0f;
    float gz_rad = gz * M_PI / 180.0f;

    // 2. Gyro Integration (Quaternion derivative)
    float dqw = 0.5f * (-qx * gx_rad - qy * gy_rad - qz * gz_rad);
    float dqx = 0.5f * ( qw * gx_rad + qy * gz_rad - qz * gy_rad);
    float dqy = 0.5f * ( qw * gy_rad - qx * gz_rad + qz * gx_rad);
    float dqz = 0.5f * ( qw * gz_rad + qx * gy_rad - qy * gx_rad);

    qw += dqw * dt;
    qx += dqx * dt;
    qy += dqy * dt;
    qz += dqz * dt;

    // 3. Normalization (Keeps the Quaternion valid)
    float norm = sqrt(qw*qw + qx*qx + qy*qy + qz*qz);
    qw /= norm; qx /= norm; qy /= norm; qz /= norm;

    // 4. Accelerometer Correction (Tilt compensation)
    // We use the Accel to fix Pitch and Roll drift
    float accel_norm = sqrt(ax*ax + ay*ay + az*az);
    if (accel_norm > 0.1f) {
        float target_pitch = atan2(-ax, sqrt(ay*ay + az*az));
        float target_roll  = atan2(ay, az);
        
        // Basic "Tilt" Quaternion from Accel
        float cp = cos(target_pitch * 0.5f);
        float sp = sin(target_pitch * 0.5f);
        float cr = cos(target_roll * 0.5f);
        float sr = sin(target_roll * 0.5f);

        float aqw = cr * cp;
        float aqx = sr * cp;
        float aqy = cr * sp;
        float aqz = -sr * sp;

        // "Nudge" the current quaternion toward the accelerometer-based quaternion
        // This is a simple Linear Interpolation (LERP) for quaternions
        float beta = 0.02f; // Influence of Accel (Drift correction)
        qw = (1.0f - beta) * qw + beta * aqw;
        qx = (1.0f - beta) * qx + beta * aqx;
        qy = (1.0f - beta) * qy + beta * aqy;
        qz = (1.0f - beta) * qz + beta * aqz;
    }

    // Stream to Serial
    Serial.printf("%lu,%.4f,%.4f,%.4f,%.4f\n", millis(), qw, qx, qy, qz);
    
    delay(10); // ~100Hz loop
}