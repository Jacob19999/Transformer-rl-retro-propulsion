// ESP32-CAM (AI-Thinker, OV2640) -> UART0: JPEG streamer and on-board optical flow sensor.
//
// Two modes, switchable at runtime ('M' command); the board boots in STREAM mode.
//   STREAM  JPEG frames for viewing.
//   FLOW    240x176 grayscale, block-matching optical flow on every frame (block_match.h, after
//           qqqlab/ESP32-Optical-Flow). Sends one small FLW1 packet per frame plus a JPEG preview
//           every `preview_interval_ms` (default 500 ms) so the portal can show what the camera sees.
//
// Packets to the PC (little-endian, each ends in CRC-16/XMODEM, Python: binascii.crc_hqx(data, 0)):
//   CAM1 | u32 jpeg_len | u32 seq | u32 millis | JPEG | u16 crc(JPEG)
//   FLW1 | u32 seq | u32 t_us | u32 dt_us | i16 dx_q8 | i16 dy_q8 | u16 sad | u16 tex | u16 compute_us | u16 crc
//          t_us   capture time of the newer frame, us since boot (low 32 bits), from the camera driver
//          dt_us  capture-time gap to the previous processed frame
//          dx/dy  image shift between the two frames in 1/256 px: a feature at (x,y) in the older frame
//                 is at (x+dx, y+dy) in the newer one
//          sad    min sum-of-abs-differences per sampled pixel x16 (match error; lower is better)
//          tex    mean abs deviation of the block x16 (image texture; low = nothing to track)
//          crc    over the bytes after the magic, before the crc
//   MSG1 | u16 len | text | u16 crc(len bytes + text)     device log lines
//
// Commands from the PC:
//   'C' framesize quality      STREAM camera settings (framesize_t; 5=QVGA 320x240, 6=CIF, 8=VGA, 3=HQVGA 240x176)
//   'M' mode                   0 = STREAM, 1 = FLOW
//   'V' u16 interval_ms        FLOW preview interval, 0 = no preview
//   'E' auto u16 aec u8 gain   exposure/gain: auto=1 automatic; auto=0 manual (aec 0..1200, gain 0..30)
//   'T'                        run the block-matcher self-test on synthetic images; results come back as MSG1
//   'B' u32 baud               switch UART speed after the current frame
//   'K' 0xA5                   keepalive; without one for 3 s at a non-boot baud the UART reverts to STREAM_BAUD
//
// Boot-ROM text at 115200 baud shows up as garbage at STREAM_BAUD; the receiver resyncs on the magic.
#include <Arduino.h>
#include <math.h>
#include <stdarg.h>
#include "esp_camera.h"
#include "esp_log.h"
#include "img_converters.h"
#include "block_match.h"

static const uint32_t STREAM_BAUD = 921600;
static const framesize_t FLOW_FRAMESIZE = FRAMESIZE_HQVGA;  // 240x176, as in the reference
static const int FLOW_WARMUP_FRAMES = 10;                   // let auto-exposure settle before reporting flow

// AI-Thinker ESP32-CAM pin map. XCLK is GPIO0, so the IO0-GND flashing jumper must be removed to run.
#define PWDN_GPIO_NUM 32
#define RESET_GPIO_NUM -1
#define XCLK_GPIO_NUM 0
#define SIOD_GPIO_NUM 26
#define SIOC_GPIO_NUM 27
#define Y9_GPIO_NUM 35
#define Y8_GPIO_NUM 34
#define Y7_GPIO_NUM 39
#define Y6_GPIO_NUM 36
#define Y5_GPIO_NUM 21
#define Y4_GPIO_NUM 19
#define Y3_GPIO_NUM 18
#define Y2_GPIO_NUM 5
#define VSYNC_GPIO_NUM 25
#define HREF_GPIO_NUM 23
#define PCLK_GPIO_NUM 22
#define LED_RED_GPIO 33  // active low

enum Mode : uint8_t { MODE_STREAM = 0, MODE_FLOW = 1 };
static Mode mode = MODE_STREAM;
static Mode requested_mode = MODE_STREAM;
static bool want_selftest = false;

// Camera settings that survive a re-init.
static int cur_framesize = FRAMESIZE_QVGA;
static int cur_quality = 12;
static bool exp_auto = true;
static uint16_t exp_value = 300;
static uint8_t gain_value = 8;
static uint16_t preview_interval_ms = 500;

// ---------------------------------------------------------------------------------------------
// Packet output
// ---------------------------------------------------------------------------------------------
static uint16_t crc16_xmodem(const uint8_t *data, size_t len, uint16_t crc = 0) {
  while (len--) {
    crc ^= (uint16_t)(*data++) << 8;
    for (int i = 0; i < 8; i++) crc = (crc & 0x8000) ? (crc << 1) ^ 0x1021 : (crc << 1);
  }
  return crc;
}

static void send_jpeg(const uint8_t *buf, size_t len) {
  static uint32_t seq = 0;
  uint32_t hdr[3] = {(uint32_t)len, seq++, millis()};
  uint16_t crc = crc16_xmodem(buf, len);
  Serial.write((const uint8_t *)"CAM1", 4);
  Serial.write((const uint8_t *)hdr, sizeof(hdr));
  Serial.write(buf, len);
  Serial.write((const uint8_t *)&crc, sizeof(crc));
}

static void send_msg(const char *fmt, ...) {
  char text[160];
  va_list ap;
  va_start(ap, fmt);
  int n = vsnprintf(text, sizeof(text), fmt, ap);
  va_end(ap);
  if (n < 0) return;
  if (n >= (int)sizeof(text)) n = sizeof(text) - 1;
  uint16_t len = (uint16_t)n;
  uint16_t crc = crc16_xmodem((const uint8_t *)&len, 2);
  crc = crc16_xmodem((const uint8_t *)text, len, crc);
  Serial.write((const uint8_t *)"MSG1", 4);
  Serial.write((const uint8_t *)&len, 2);
  Serial.write((const uint8_t *)text, len);
  Serial.write((const uint8_t *)&crc, 2);
}

struct __attribute__((packed)) FlowPacket {
  char magic[4];
  uint32_t seq;
  uint32_t t_us;
  uint32_t dt_us;
  int16_t dx_q8;
  int16_t dy_q8;
  uint16_t sad;
  uint16_t tex;
  uint16_t compute_us;
  uint16_t crc;
};

static int16_t to_q8(float v) {
  long q = lroundf(v * 256.0f);
  return (int16_t)(q > 32767 ? 32767 : (q < -32768 ? -32768 : q));
}

static uint16_t sat_u16(uint32_t v) { return (uint16_t)(v > 65535 ? 65535 : v); }

// ---------------------------------------------------------------------------------------------
// Camera
// ---------------------------------------------------------------------------------------------
static void apply_settings() {
  sensor_t *s = esp_camera_sensor_get();
  if (!s) return;
  if (mode == MODE_STREAM) {
    s->set_framesize(s, (framesize_t)cur_framesize);
    s->set_quality(s, cur_quality);
  }
  s->set_exposure_ctrl(s, exp_auto);
  s->set_gain_ctrl(s, exp_auto);
  if (!exp_auto) {
    s->set_aec_value(s, exp_value);
    s->set_agc_gain(s, gain_value);
  }
}

static bool init_camera() {
  camera_config_t c = {};
  c.ledc_channel = LEDC_CHANNEL_0;
  c.ledc_timer = LEDC_TIMER_0;
  c.pin_d0 = Y2_GPIO_NUM;
  c.pin_d1 = Y3_GPIO_NUM;
  c.pin_d2 = Y4_GPIO_NUM;
  c.pin_d3 = Y5_GPIO_NUM;
  c.pin_d4 = Y6_GPIO_NUM;
  c.pin_d5 = Y7_GPIO_NUM;
  c.pin_d6 = Y8_GPIO_NUM;
  c.pin_d7 = Y9_GPIO_NUM;
  c.pin_xclk = XCLK_GPIO_NUM;
  c.pin_pclk = PCLK_GPIO_NUM;
  c.pin_vsync = VSYNC_GPIO_NUM;
  c.pin_href = HREF_GPIO_NUM;
  c.pin_sccb_sda = SIOD_GPIO_NUM;
  c.pin_sccb_scl = SIOC_GPIO_NUM;
  c.pin_pwdn = PWDN_GPIO_NUM;
  c.pin_reset = RESET_GPIO_NUM;
  c.xclk_freq_hz = 20000000;
  c.grab_mode = CAMERA_GRAB_LATEST;
  if (mode == MODE_FLOW) {
    c.pixel_format = PIXFORMAT_GRAYSCALE;
    c.frame_size = FLOW_FRAMESIZE;
    c.fb_count = 3;  // one held as the "previous" frame, one being processed, one being filled by DMA
    c.fb_location = CAMERA_FB_IN_DRAM;  // ARPS reads every frame many times; PSRAM is much slower
  } else {
    c.pixel_format = PIXFORMAT_JPEG;
    // JPEG frame buffers are sized from the init framesize, so init at the largest size we allow and
    // scale down afterwards. Initialising at QVGA and switching up to VGA overflows and wedges the driver.
    c.frame_size = psramFound() ? FRAMESIZE_UXGA : FRAMESIZE_QVGA;
    c.jpeg_quality = 12;
    c.fb_count = psramFound() ? 2 : 1;
    c.fb_location = psramFound() ? CAMERA_FB_IN_PSRAM : CAMERA_FB_IN_DRAM;
  }
  if (esp_camera_init(&c) != ESP_OK) return false;
  apply_settings();
  return true;
}

static void blink_forever(int period_ms) {
  pinMode(LED_RED_GPIO, OUTPUT);
  for (;;) {
    digitalWrite(LED_RED_GPIO, LOW);
    delay(period_ms);
    digitalWrite(LED_RED_GPIO, HIGH);
    delay(period_ms);
  }
}

// Flow state
static camera_fb_t *fb_last = nullptr;
static int8_t seed_dx = 0, seed_dy = 0;
static uint32_t flow_seq = 0;
static int64_t prev_t_us = 0;
static int warmup = 0;
static uint32_t last_preview_ms = 0;
static bm::Workspace ws;

static void release_last_frame() {
  if (fb_last) {
    esp_camera_fb_return(fb_last);
    fb_last = nullptr;
  }
}

static bool start_mode(Mode m) {
  release_last_frame();
  esp_camera_deinit();
  mode = m;
  if (!init_camera()) return false;
  seed_dx = seed_dy = 0;
  flow_seq = 0;
  warmup = FLOW_WARMUP_FRAMES;
  last_preview_ms = 0;
  return true;
}

static void switch_mode(Mode m) {
  if (start_mode(m)) {
    if (m == MODE_FLOW)
      send_msg("mode=FLOW %dx%d gray, 3 fb in DRAM, heap free %u", 240, 176, (unsigned)ESP.getFreeHeap());
    else
      send_msg("mode=STREAM jpeg, heap free %u", (unsigned)ESP.getFreeHeap());
    return;
  }
  send_msg("mode %d init FAILED (heap free %u); falling back to STREAM", (int)m, (unsigned)ESP.getFreeHeap());
  requested_mode = MODE_STREAM;
  if (m != MODE_STREAM && start_mode(MODE_STREAM)) return;
  blink_forever(100);  // fast blink = camera init failed
}

// ---------------------------------------------------------------------------------------------
// Self-test: run the block matcher on synthetic textures shifted by known sub-pixel amounts.
// ---------------------------------------------------------------------------------------------
static float lattice(int ix, int iy, uint32_t seed) {
  uint32_t h = (uint32_t)ix * 374761393u + (uint32_t)iy * 668265263u + seed * 2246822519u;
  h = (h ^ (h >> 13)) * 1274126177u;
  h ^= h >> 16;
  return (h & 0xFFFF) / 65535.0f;
}

static float value_noise(float x, float y, uint32_t seed) {
  int ix = (int)floorf(x), iy = (int)floorf(y);
  float fx = x - ix, fy = y - iy;
  fx = fx * fx * (3 - 2 * fx);
  fy = fy * fy * (3 - 2 * fy);
  float a = lattice(ix, iy, seed), b = lattice(ix + 1, iy, seed);
  float c = lattice(ix, iy + 1, seed), d = lattice(ix + 1, iy + 1, seed);
  return a + (b - a) * fx + (c - a) * fy + (a - b - c + d) * fx * fy;
}

static uint8_t texture_at(float x, float y) {
  float v = 0.6f * value_noise(x / 6, y / 6, 1) + 0.3f * value_noise(x / 13, y / 13, 2) +
            0.1f * value_noise(x / 3, y / 3, 3);
  v = (v - 0.25f) / 0.5f;  // value noise clusters mid-range; stretch to use the dynamic range
  return (uint8_t)(v < 0 ? 0 : (v > 1 ? 255 : v * 255));
}

static void run_selftest() {
  const int w = 240, h = 176;
  uint8_t *a = (uint8_t *)heap_caps_malloc(w * h, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
  uint8_t *b = (uint8_t *)heap_caps_malloc(w * h, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
  if (!a || !b) {
    send_msg("selftest: cannot allocate 2x%d bytes in DRAM (free %u); try in STREAM mode", w * h,
             (unsigned)ESP.getFreeHeap());
    free(a);
    free(b);
    return;
  }
  const float shifts[][2] = {{0, 0}, {1, 0}, {0.5f, 0.5f}, {3.4f, -2.6f}, {-1.25f, 0.75f}, {-7.25f, 5.5f}, {12.3f, -11.7f}};
  send_msg("selftest: ARPS P=%d step=%d on %dx%d synthetic texture (seed reset each case)", bm::PMAX, bm::BSTEP, w, h);
  for (auto &s : shifts) {
    for (int y = 0; y < h; y++)
      for (int x = 0; x < w; x++) {
        a[y * w + x] = texture_at(x, y);
        b[y * w + x] = texture_at(x - s[0], y - s[1]);  // content moved by (+s0, +s1)
      }
    int8_t sdx = 0, sdy = 0;
    uint32_t t0 = micros();
    bm::Result r = bm::arps(ws, a, b, w, h, sdx, sdy);
    uint32_t us = micros() - t0;
    send_msg("true(%+.2f,%+.2f) est(%+.2f,%+.2f) err(%+.2f,%+.2f) int(%d,%d) sad=%.2f tex=%.1f evals=%u %uus",
             s[0], s[1], r.sub_dx, r.sub_dy, r.sub_dx - s[0], r.sub_dy - s[1], r.dx, r.dy, r.sad_x16 / 16.0f,
             r.tex_x16 / 16.0f, (unsigned)r.evals, (unsigned)us);
  }
  free(a);
  free(b);
  send_msg("selftest: done");
}

// ---------------------------------------------------------------------------------------------
// Per-frame work
// ---------------------------------------------------------------------------------------------
static int null_frames = 0;

// Camera stopped delivering: re-initialise it in the current mode with the current settings.
static bool handle_null_frame() {
  if (++null_frames < 3) return false;
  null_frames = 0;
  if (!start_mode(mode)) blink_forever(100);
  send_msg("camera stalled; re-initialised");
  return true;
}

static void stream_step() {
  camera_fb_t *fb = esp_camera_fb_get();
  if (!fb) {
    handle_null_frame();
    return;
  }
  null_frames = 0;
  send_jpeg(fb->buf, fb->len);
  esp_camera_fb_return(fb);
  static uint32_t n = 0;
  digitalWrite(LED_RED_GPIO, (++n & 8) ? HIGH : LOW);  // slow blink = frames flowing
}

static void flow_step() {
  camera_fb_t *fb = esp_camera_fb_get();
  if (!fb) {
    handle_null_frame();
    return;
  }
  null_frames = 0;
  if (fb->format != PIXFORMAT_GRAYSCALE) {  // not what we configured; do not feed it to the matcher
    esp_camera_fb_return(fb);
    return;
  }
  int64_t t_us = (int64_t)fb->timestamp.tv_sec * 1000000LL + fb->timestamp.tv_usec;

  if (fb_last && warmup == 0) {
    uint32_t t0 = micros();
    bm::Result r = bm::arps(ws, fb_last->buf, fb->buf, fb->width, fb->height, seed_dx, seed_dy);
    uint32_t compute_us = micros() - t0;
    FlowPacket p;
    memcpy(p.magic, "FLW1", 4);
    p.seq = flow_seq++;
    p.t_us = (uint32_t)t_us;
    p.dt_us = (uint32_t)(t_us - prev_t_us);
    p.dx_q8 = to_q8(r.sub_dx);
    p.dy_q8 = to_q8(r.sub_dy);
    p.sad = sat_u16(r.sad_x16);
    p.tex = sat_u16(r.tex_x16);
    p.compute_us = sat_u16(compute_us);
    p.crc = crc16_xmodem((const uint8_t *)&p + 4, sizeof(p) - 6);
    Serial.write((const uint8_t *)&p, sizeof(p));
  }
  if (warmup) warmup--;
  release_last_frame();
  fb_last = fb;  // held until the next frame arrives: it is the "previous" frame for the next match
  prev_t_us = t_us;

  if (preview_interval_ms && millis() - last_preview_ms >= preview_interval_ms) {
    last_preview_ms = millis();
    uint8_t *jpg = nullptr;
    size_t jpg_len = 0;
    if (frame2jpg(fb, 30, &jpg, &jpg_len)) {
      send_jpeg(jpg, jpg_len);
      free(jpg);
    }
  }
  digitalWrite(LED_RED_GPIO, (flow_seq & 16) ? HIGH : LOW);
}

// ---------------------------------------------------------------------------------------------
// Commands
// ---------------------------------------------------------------------------------------------
static uint32_t current_baud = STREAM_BAUD;
static uint32_t last_keepalive_ms = 0;
static const uint32_t KEEPALIVE_TIMEOUT_MS = 3000;

static int command_length(int head) {
  switch (head) {
    case 'K': return 2;
    case 'C': return 3;
    case 'M': return 2;
    case 'V': return 3;
    case 'E': return 5;
    case 'T': return 1;
    case 'B': return 5;
    default: return 0;  // not a command byte
  }
}

static void handle_commands() {
  for (;;) {
    int head = Serial.peek();
    if (head < 0) return;
    int need = command_length(head);
    if (!need) {
      Serial.read();  // noise: discard byte-by-byte
      continue;
    }
    if (Serial.available() < need) return;  // partial command; wait for the rest
    uint8_t cmd[5];
    for (int i = 0; i < need; i++) cmd[i] = Serial.read();
    switch (head) {
      case 'K':
        if (cmd[1] == 0xA5) last_keepalive_ms = millis();
        break;
      case 'C':
        if (cmd[1] >= FRAMESIZE_QQVGA && cmd[1] <= FRAMESIZE_SXGA) cur_framesize = cmd[1];
        if (cmd[2] >= 4 && cmd[2] <= 63) cur_quality = cmd[2];
        apply_settings();
        break;
      case 'M':
        if (cmd[1] <= MODE_FLOW) requested_mode = (Mode)cmd[1];
        break;
      case 'V':
        preview_interval_ms = cmd[1] | (cmd[2] << 8);
        break;
      case 'E':
        exp_auto = cmd[1] != 0;
        exp_value = min(1200, cmd[2] | (cmd[3] << 8));
        gain_value = min(30, (int)cmd[4]);
        apply_settings();
        break;
      case 'T':
        want_selftest = true;
        break;
      case 'B': {
        uint32_t baud = cmd[1] | (cmd[2] << 8) | (cmd[3] << 16) | ((uint32_t)cmd[4] << 24);
        if (baud >= 115200 && baud <= 4000000) {
          Serial.flush();  // let queued TX finish at the old rate
          Serial.updateBaudRate(baud);
          current_baud = baud;
          last_keepalive_ms = millis();
        }
        break;
      }
    }
  }
}

void setup() {
  Serial.begin(STREAM_BAUD);
  Serial.setDebugOutput(false);
  esp_log_level_set("*", ESP_LOG_NONE);  // keep IDF log text out of the binary stream
  if (!start_mode(MODE_STREAM)) blink_forever(100);  // fast blink = camera init failed
  pinMode(LED_RED_GPIO, OUTPUT);
  digitalWrite(LED_RED_GPIO, HIGH);
}

void loop() {
  handle_commands();
  if (current_baud != STREAM_BAUD && millis() - last_keepalive_ms > KEEPALIVE_TIMEOUT_MS) {
    Serial.flush();
    Serial.updateBaudRate(STREAM_BAUD);  // PC lost or never followed the switch: fall back to the boot rate
    current_baud = STREAM_BAUD;
  }
  if (requested_mode != mode) switch_mode(requested_mode);
  if (want_selftest) {
    want_selftest = false;
    run_selftest();
  }
  if (mode == MODE_FLOW) flow_step(); else stream_step();
}
