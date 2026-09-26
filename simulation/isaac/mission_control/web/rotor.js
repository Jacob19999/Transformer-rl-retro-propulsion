// Real EDF speed is hundreds of revolutions/s. Slow the visualization 200x
// to make RPM changes visible at display frame rates.
// The telemetry readout remains actual RPM.
export const ROTOR_VISUAL_SPEED_SCALE = 1 / 200;
const TAU = 2 * Math.PI;
const rpmToRadians = TAU / 60 * ROTOR_VISUAL_SPEED_SCALE;
const rpm = frame => Number.isFinite(frame?.rotor_rpm) ? Math.max(0, frame.rotor_rpm) : 0;

export function createRotorAnimation(rotor) {
  let phases = new Float64Array(0);
  return {
    setFrames(frames) {
      phases = new Float64Array(frames.length);
      for (let i = 1; i < frames.length; i++) {
        const dt = Math.max(0, frames[i].t - frames[i - 1].t);
        phases[i] = (phases[i - 1] + dt * (rpm(frames[i - 1]) + rpm(frames[i])) / 2 * rpmToRadians) % TAU;
      }
    },
    update(sample) {
      // Older cached CAD exports have no separate EDFRotor node. The fan is
      // optional visual detail: its absence must not stop all four cameras.
      if (!rotor) return;
      if (!sample) { rotor.rotation.z = 0; return; }
      const {a, b, alpha, index} = sample;
      const dt = Math.max(0, b.t - a.t);
      // Integrate linearly interpolated RPM, including spool-up/coast-down.
      // An absolute timeline phase makes seeks, pause, live frame appends,
      // replay speed changes and exported video independent of render rate.
      const partial = dt * (rpm(a) * alpha + (rpm(b) - rpm(a)) * alpha * alpha / 2);
      rotor.rotation.z = ((phases[index] ?? 0) + partial * rpmToRadians) % TAU;
    },
  };
}
