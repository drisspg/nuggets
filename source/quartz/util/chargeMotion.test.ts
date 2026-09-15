import assert from "node:assert/strict"
import { test } from "node:test"
import { chargeMotion } from "./chargeMotion"

test("like charges turn around without crossing", () => {
  assert.deepEqual(chargeMotion(0), { separation: 1, velocity: 0 })
  for (const time of [0.001, 0.1, 0.5, 1, 2, 4]) {
    const incoming = chargeMotion(-time)
    const outgoing = chargeMotion(time)
    assert(incoming.velocity < 0)
    assert(outgoing.velocity > 0)
    assert(outgoing.separation > 1)
    assert.equal(incoming.separation, outgoing.separation)
    assert.equal(incoming.velocity, -outgoing.velocity)
  }
})

test("Coulomb energy is conserved through the encounter", () => {
  for (let i = -80; i <= 80; i++) {
    const { separation, velocity } = chargeMotion(i / 20)
    // Double precision arithmetic with ample allowance for the Newton solve.
    assert(Math.abs(velocity ** 2 / 4 + 1 / separation - 1) < 1e-12)
  }
})

test("motion follows inverse-square repulsion rather than sinusoidal easing", () => {
  const dt = 1e-4
  for (const time of [-4, -2, -0.5, 0, 0.5, 2, 4]) {
    const before = chargeMotion(time - dt)
    const after = chargeMotion(time + dt)
    const center = chargeMotion(time)
    const velocity = (after.separation - before.separation) / (2 * dt)
    const acceleration = (after.velocity - before.velocity) / (2 * dt)
    // Central differences have O(dt²) truncation error; 1e-6 leaves margin.
    assert(Math.abs(velocity - center.velocity) < 1e-6)
    assert(Math.abs(acceleration - 2 / center.separation ** 2) < 1e-6)
  }
})

test("analytic trajectory agrees with independent velocity-Verlet integration", () => {
  const dt = 0.001
  let separation = 1
  let velocity = 0
  for (let i = 1; i <= 4000; i++) {
    const acceleration = 2 / separation ** 2
    separation += velocity * dt + 0.5 * acceleration * dt ** 2
    velocity += 0.5 * (acceleration + 2 / separation ** 2) * dt
    if (i % 1000 === 0) {
      const exact = chargeMotion(i * dt)
      // Verlet's global error is O(dt²); budget 1e-5 over four time units.
      assert(Math.abs(separation - exact.separation) < 1e-5)
      assert(Math.abs(velocity - exact.velocity) < 1e-5)
    }
  }
})
