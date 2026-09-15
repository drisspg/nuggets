// Classical, head-on scattering of two equal like charges. Dimensionless units:
// each mass = 1, Coulomb constant * charge² = 1, total energy = 1.
// Relative separation obeys r'' = 2/r² and E = r'²/4 + 1/r.
// With r = cosh²(u), t = (u + sinh(u)cosh(u))/2; t=0 is closest approach.
// This models the packet centers, not electron wavefunctions or radiation.
export function chargeMotion(time: number) {
  const target = Math.abs(time)
  let u = Math.asinh(Math.sqrt(2 * target))
  for (let i = 0; i < 8; i++) {
    const cosh = Math.cosh(u)
    u -= ((u + Math.sinh(u) * cosh) / 2 - target) / (cosh * cosh)
  }
  return {
    separation: Math.cosh(u) ** 2,
    velocity: Math.sign(time) * 2 * Math.tanh(u),
  }
}
