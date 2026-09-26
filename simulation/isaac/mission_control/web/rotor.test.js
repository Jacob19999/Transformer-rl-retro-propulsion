import test from 'node:test';
import assert from 'node:assert/strict';
import { createRotorAnimation } from './rotor.js';

function setup(frames) {
  const rotor = {rotation: {z: 0}};
  const animation = createRotorAnimation(rotor);
  animation.setFrames(frames);
  return {rotor, animation};
}
// Zero and 2π are the same orientation, including floating-point wraparound.
const near = (actual, expected) => assert.ok(
  Math.abs(Math.atan2(Math.sin(actual - expected), Math.cos(actual - expected))) < 1e-10,
  `${actual} != ${expected}`,
);

test('legacy models without a rotor do not interrupt startup, replay or seeking', () => {
  const frames = [{t: 0, rotor_rpm: 0}, {t: 1, rotor_rpm: 30000}];
  for (const rotor of [undefined, null]) {
    const animation = createRotorAnimation(rotor);
    assert.doesNotThrow(() => animation.update(null));
    animation.setFrames(frames);
    assert.doesNotThrow(() => animation.update({a: frames[0], b: frames[1], index: 0, alpha: .5}));
    assert.doesNotThrow(() => animation.update({a: frames[1], b: frames[1], index: 1, alpha: 0}));
    assert.doesNotThrow(() => animation.update({a: frames[0], b: frames[1], index: 0, alpha: 0}));
  }
});

test('30,000 RPM gives 2.5 visible revolutions/s; seeks and redraws preserve phase', () => {
  const frames = [{t: 0, rotor_rpm: 30000}, {t: 2, rotor_rpm: 30000}];
  const {rotor, animation} = setup(frames);
  const sample = {a: frames[0], b: frames[1], index: 0, alpha: .0625};
  animation.update(sample);
  near(rotor.rotation.z, .625 * Math.PI);
  animation.update({...sample, alpha: .375});
  near(rotor.rotation.z, 1.75 * Math.PI);
  animation.update(sample);
  near(rotor.rotation.z, .625 * Math.PI);
  animation.update(sample);
  near(rotor.rotation.z, .625 * Math.PI);
});

test('spool-up and coast-down integrate changing RPM; stopped rotor holds phase', () => {
  const frames = [
    {t: 0, rotor_rpm: 0}, {t: 1, rotor_rpm: 30000},
    {t: 2, rotor_rpm: 0}, {t: 3, rotor_rpm: 0},
  ];
  const {rotor, animation} = setup(frames);
  animation.update({a: frames[0], b: frames[1], index: 0, alpha: .5});
  near(rotor.rotation.z, .625 * Math.PI);
  animation.update({a: frames[1], b: frames[2], index: 1, alpha: .5});
  near(rotor.rotation.z, .375 * Math.PI);
  animation.update({a: frames[2], b: frames[3], index: 2, alpha: .75});
  near(rotor.rotation.z, Math.PI);
  animation.update({a: frames[3], b: frames[3], index: 3, alpha: 0});
  near(rotor.rotation.z, Math.PI);
});

test('live frame appends preserve phase; mission replacement resets it', () => {
  const frames = [{t: 0, rotor_rpm: 30000}, {t: .125, rotor_rpm: 30000}];
  const {rotor, animation} = setup(frames);
  animation.update({a: frames[1], b: frames[1], index: 1, alpha: 0});
  near(rotor.rotation.z, .625 * Math.PI);
  frames.push({t: .25, rotor_rpm: 30000});
  animation.setFrames(frames);
  animation.update({a: frames[1], b: frames[2], index: 1, alpha: 0});
  near(rotor.rotation.z, .625 * Math.PI);
  animation.setFrames([{t: 0, rotor_rpm: 0}]);
  animation.update(null);
  near(rotor.rotation.z, 0);
});

test('missing, invalid and negative RPM do not fabricate spin or poison rotation', () => {
  const frames = [{t: 0}, {t: 1, rotor_rpm: NaN}, {t: 2, rotor_rpm: -200}, {t: 3, rotor_rpm: 0}];
  const {rotor, animation} = setup(frames);
  animation.update({a: frames[2], b: frames[3], index: 2, alpha: .5});
  near(rotor.rotation.z, 0);
});
