import assert from "node:assert/strict"
import { test } from "node:test"
import { Charge, CHARGE_SOFTENING, COULOMB_STRENGTH, stepCharges } from "./chargeMotion"

const pair = (sign: 1 | -1): Charge[] => [
  { x: -0.7, y: 0, vx: 0, vy: -0.2, charge: 1 },
  { x: 0.7, y: 0, vx: 0, vy: 0.2, charge: sign },
]

function energy(charges: Charge[]) {
  let total = charges.reduce((sum, p) => sum + (p.vx ** 2 + p.vy ** 2) / 2, 0)
  for (let i = 0; i < charges.length; i++) {
    for (let j = i + 1; j < charges.length; j++) {
      const a = charges[i]
      const b = charges[j]
      const radius = Math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2 + CHARGE_SOFTENING ** 2)
      total += (COULOMB_STRENGTH * a.charge * b.charge) / radius
    }
  }
  return total
}

test("like signs repel and opposite signs attract", () => {
  for (const sign of [1, -1] as const) {
    const charges = pair(sign)
    stepCharges(charges, 0.01)
    assert(charges[0].vx * sign < 0)
    assert(charges[1].vx * sign > 0)
    assert(Math.abs(charges[0].vx + charges[1].vx) < 1e-12)
  }
  const negatives = pair(1).map((p) => ({ ...p, charge: -1 as const }))
  stepCharges(negatives, 0.01)
  assert(negatives[0].vx < 0 && negatives[1].vx > 0)
})

test("isolated charges travel freely and an empty scene is valid", () => {
  const charges: Charge[] = [{ x: 1, y: 2, vx: 3, vy: -4, charge: 1 }]
  stepCharges(charges, 0.5)
  assert.deepEqual(charges[0], { x: 2.5, y: 0, vx: 3, vy: -4, charge: 1 })
  stepCharges([], 0.5)
})

test("softening keeps overlapping and close charges finite", () => {
  for (const offset of [0, 0.001, 0.1]) {
    const charges = pair(-1)
    charges[0].x = -offset
    charges[1].x = offset
    for (let i = 0; i < 1000; i++) stepCharges(charges, 1 / 240)
    assert(charges.every((p) => [p.x, p.y, p.vx, p.vy].every(Number.isFinite)))
  }
})

test("a closed mixed-charge system conserves momentum and bounded energy", () => {
  const charges = pair(-1)
  charges.push({ x: 2, y: 1, vx: -0.1, vy: 0.1, charge: -1 })
  const initialEnergy = energy(charges)
  const initialAngularMomentum = charges.reduce((sum, p) => sum + p.x * p.vy - p.y * p.vx, 0)
  for (let i = 0; i < 5000; i++) {
    stepCharges(charges, 1 / 240)
    assert(Math.abs(charges.reduce((sum, p) => sum + p.vx, 0) + 0.1) < 1e-10)
    assert(Math.abs(charges.reduce((sum, p) => sum + p.vy, 0) - 0.1) < 1e-10)
    const angularMomentum = charges.reduce((sum, p) => sum + p.x * p.vy - p.y * p.vx, 0)
    assert(Math.abs(angularMomentum - initialAngularMomentum) < 1e-10)
    // Fixed-step Verlet is second order. Allow 0.2% of the unit energy scale
    // for the bounded oscillatory error during close softened encounters.
    assert(Math.abs(energy(charges) - initialEnergy) < 0.002)
  }
})

test("halving the timestep converges toward a finer integration", () => {
  const integrate = (dt: number) => {
    const charges = pair(-1)
    for (let i = 0; i < Math.round(2 / dt); i++) stepCharges(charges, dt)
    return charges[0]
  }
  const coarse = integrate(1 / 120)
  const fine = integrate(1 / 240)
  const reference = integrate(1 / 1920)
  const error = (p: Charge) => Math.hypot(p.x - reference.x, p.y - reference.y)
  assert(error(fine) < error(coarse) / 3)
})
