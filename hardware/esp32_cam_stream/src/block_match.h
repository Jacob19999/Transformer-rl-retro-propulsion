// Block-matching optical flow for 8-bit grayscale frames. No Arduino/ESP dependencies.
//
// Algorithm and structure follow qqqlab/ESP32-Optical-Flow (MIT, (c) 2024 qqqlab):
// one whole-frame block, Adaptive Rood Pattern Search (ARPS) on the Sum of Absolute Differences,
// the previous frame's motion as the search seed, and a subsampled block (BSTEP) for speed.
// Additions over the reference: every SAD evaluated is kept, so the final integer minimum is refined
// to sub-pixel with the two neighbouring SAD values (free, no extra evaluations), and match-quality
// metrics (min SAD per sample, image texture) are returned.
//
// Convention: a feature at (x, y) in `prev` is found at (x + dx, y + dy) in `cur`.
#pragma once
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

namespace bm {

constexpr int PMAX = 15;   // maximum search distance in pixels; also the frame border that is excluded
constexpr int BSTEP = 2;   // sample every BSTEP-th pixel of the block in x and y

struct Result {
  int8_t dx, dy;         // integer displacement of the SAD minimum
  float sub_dx, sub_dy;  // refined displacement (pixels)
  uint32_t sad_x16;      // min SAD per sampled pixel, x16
  uint32_t tex_x16;      // mean absolute deviation of the sampled block in `cur`, x16
  uint16_t evals;        // number of SAD evaluations used
};

struct Workspace {
  bool done[2 * PMAX + 1][2 * PMAX + 1];
  uint32_t sad[2 * PMAX + 1][2 * PMAX + 1];
};

inline uint32_t sad_block(const uint8_t *a, const uint8_t *b, int w, int bx, int by, int bw, int bh,
                          int dx, int dy) {
  uint32_t sum = 0;
  for (int y = by; y < by + bh; y += BSTEP) {
    const uint8_t *pa = a + y * w + bx;
    const uint8_t *pb = b + (y + dy) * w + bx + dx;
    for (int x = 0; x < bw; x += BSTEP) {
      int d = (int)*pa - (int)*pb;
      sum += d < 0 ? -d : d;
      pa += BSTEP;
      pb += BSTEP;
    }
  }
  return sum;
}

// Mean absolute deviation of the sampled block: how much structure there is to track.
inline uint32_t texture_x16(const uint8_t *buf, int w, int bx, int by, int bw, int bh) {
  uint32_t sum = 0, n = 0;
  for (int y = by; y < by + bh; y += BSTEP)
    for (int x = 0; x < bw; x += BSTEP) { sum += buf[y * w + bx + x]; n++; }
  if (!n) return 0;
  int avg = (sum + n / 2) / n;
  uint32_t dev = 0;
  for (int y = by; y < by + bh; y += BSTEP)
    for (int x = 0; x < bw; x += BSTEP) { int d = (int)buf[y * w + bx + x] - avg; dev += d < 0 ? -d : d; }
  return (dev * 16 + n / 2) / n;
}

// Sub-pixel offset of a minimum from its two neighbours. SAD is V-shaped near the optimum
// (not parabolic), so a symmetric-V fit is used: offset in [-0.5, 0.5].
inline float v_fit(uint32_t s_minus, uint32_t s_zero, uint32_t s_plus) {
  int32_t a = (int32_t)s_minus - (int32_t)s_zero;
  int32_t b = (int32_t)s_plus - (int32_t)s_zero;
  int32_t m = a > b ? a : b;
  if (m <= 0) return 0.0f;
  return 0.5f * (float)(a - b) / (float)m;
}

// prev/cur: w*h grayscale. seed_dx/seed_dy: previous result, used as the search start (updated in place).
inline Result arps(Workspace &ws, const uint8_t *prev, const uint8_t *cur, int w, int h,
                   int8_t &seed_dx, int8_t &seed_dy) {
  const int p = PMAX;
  const int bx = p, by = p, bw = w - 2 * p, bh = h - 2 * p;
  memset(ws.done, 0, sizeof(ws.done));

  int min_dx = 0, min_dy = 0;
  uint32_t min_sum = 0xFFFFFFFFu;
  uint16_t evals = 0;

  auto check = [&](int dx, int dy) {
    if (dx < -p || dx > p || dy < -p || dy > p || ws.done[p + dx][p + dy]) return;
    ws.done[p + dx][p + dy] = true;
    uint32_t s = sad_block(prev, cur, w, bx, by, bw, bh, dx, dy);
    ws.sad[p + dx][p + dy] = s;
    evals++;
    if (s < min_sum) { min_sum = s; min_dx = dx; min_dy = dy; }
  };

  int sx = seed_dx < -p ? -p : (seed_dx > p ? p : seed_dx);
  int sy = seed_dy < -p ? -p : (seed_dy > p ? p : seed_dy);
  int S = abs(sx) > abs(sy) ? abs(sx) : abs(sy);
  if (S < 2) S = 2;

  check(sx, sy);  // previous motion first (addition in the reference)
  check(0, 0);
  check(+S, 0); check(-S, 0); check(0, +S); check(0, -S);

  int cx, cy;
  do {  // small "+" pattern around the current minimum until the centre stays the minimum
    cx = min_dx; cy = min_dy;
    check(cx + 1, cy); check(cx - 1, cy); check(cx, cy + 1); check(cx, cy - 1);
  } while (cx != min_dx || cy != min_dy);

  Result r;
  r.dx = (int8_t)min_dx;
  r.dy = (int8_t)min_dy;
  r.sub_dx = (float)min_dx;
  r.sub_dy = (float)min_dy;
  if (min_dx > -p && min_dx < p && ws.done[p + min_dx - 1][p + min_dy] && ws.done[p + min_dx + 1][p + min_dy])
    r.sub_dx += v_fit(ws.sad[p + min_dx - 1][p + min_dy], min_sum, ws.sad[p + min_dx + 1][p + min_dy]);
  if (min_dy > -p && min_dy < p && ws.done[p + min_dx][p + min_dy - 1] && ws.done[p + min_dx][p + min_dy + 1])
    r.sub_dy += v_fit(ws.sad[p + min_dx][p + min_dy - 1], min_sum, ws.sad[p + min_dx][p + min_dy + 1]);

  uint32_t n = (uint32_t)((bh + BSTEP - 1) / BSTEP) * (uint32_t)((bw + BSTEP - 1) / BSTEP);
  r.sad_x16 = (min_sum * 16 + n / 2) / n;
  r.tex_x16 = texture_x16(cur, w, bx, by, bw, bh);
  r.evals = evals;
  seed_dx = r.dx;
  seed_dy = r.dy;
  return r;
}

}  // namespace bm
