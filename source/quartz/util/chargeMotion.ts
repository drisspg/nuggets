export interface Charge {
  x: number
  y: number
  vx: number
  vy: number
  charge: 1 | -1
}

export const COULOMB_STRENGTH = 0.5
export const CHARGE_SOFTENING = 0.2

// Equal-mass classical charges with softened Coulomb potential
// U = k*q1*q2 / sqrt(r² + epsilon²). Softening bounds close-encounter forces;
// this is a decorative classical model, not quantum electron/positron dynamics.
export function stepCharges(charges: Charge[], dt: number) {
  const kick = () => {
    for (let i = 0; i < charges.length; i++) {
      for (let j = i + 1; j < charges.length; j++) {
        const a = charges[i]
        const b = charges[j]
        const dx = a.x - b.x
        const dy = a.y - b.y
        const radiusSquared = dx * dx + dy * dy + CHARGE_SOFTENING ** 2
        const impulse = (COULOMB_STRENGTH * a.charge * b.charge * dt) / (2 * radiusSquared ** 1.5)
        a.vx += dx * impulse
        a.vy += dy * impulse
        b.vx -= dx * impulse
        b.vy -= dy * impulse
      }
    }
  }

  // Velocity-Verlet: symmetric half kicks preserve pairwise momentum exactly.
  kick()
  for (const particle of charges) {
    particle.x += particle.vx * dt
    particle.y += particle.vy * dt
  }
  kick()
}
