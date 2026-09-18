import { Charge, stepCharges } from "../../util/chargeMotion"

type VisibleCharge = Charge & { node: SVGGElement; age: number; lifetime: number }

document.addEventListener("nav", () => {
  const svg = document.querySelector<SVGSVGElement>(".home-collision")
  const template = svg?.querySelector<SVGGElement>("#charge-packet-template")
  if (!svg || !template) return

  const particles: VisibleCharge[] = []
  let width = 12
  let height = 12
  const resize = () => {
    const bounds = svg.getBoundingClientRect()
    if (!bounds.width || !bounds.height) return
    // Match the decoration's aspect ratio so fields stay circular on every screen.
    const scale = 12 / Math.min(bounds.width, bounds.height)
    const nextWidth = bounds.width * scale
    const nextHeight = bounds.height * scale
    for (const p of particles) {
      p.x *= nextWidth / width
      p.y *= nextHeight / height
    }
    width = nextWidth
    height = nextHeight
    svg.setAttribute("viewBox", `0 0 ${width * 50} ${height * 50}`)
  }
  resize()
  const observer = new ResizeObserver(resize)
  observer.observe(svg)

  const random = (min: number, max: number) => min + Math.random() * (max - min)
  const spawn = () => {
    if (particles.length >= 3) return
    let x = 0
    let y = 0
    for (let attempt = 0; attempt < 12; attempt++) {
      x = random(width * 0.1, width * 0.9)
      y = random(height * 0.15, height * 0.35)
      if (particles.every((p) => Math.hypot(p.x - x, p.y - y) > 0.8)) break
      if (attempt === 11) return
    }
    // Keep both signs represented; additional charges have random signs.
    const charge = !particles.some((p) => p.charge === 1)
      ? 1
      : !particles.some((p) => p.charge === -1)
        ? -1
        : Math.random() < 0.5
          ? 1
          : -1
    const node = template.cloneNode(true) as SVGGElement
    node.removeAttribute("id")
    node.classList.add("charge-packet")
    node.dataset.charge = String(charge)
    node.style.opacity = "0"
    svg.append(node)
    particles.push({
      x,
      y,
      charge,
      node,
      age: 0,
      lifetime: random(24, 36),
      vx: random(-0.12, 0.12),
      vy: random(-0.12, 0.12),
    })
  }

  for (let i = 0; i < 3; i++) spawn()
  let previous = performance.now()
  let accumulator = 0
  let nextSpawn = random(2.5, 4.5)
  let frame: number
  const step = 1 / 240
  const draw = (now: number) => {
    // Do not catch up a background tab with one large, unstable physics step.
    const elapsed = Math.min((now - previous) / 1000, 0.05)
    previous = now
    nextSpawn -= elapsed
    if (nextSpawn <= 0) {
      spawn()
      nextSpawn = random(2.5, 4.5)
    }
    accumulator += elapsed * 0.45
    while (accumulator >= step) {
      stepCharges(particles, step)
      accumulator -= step
    }
    for (let i = particles.length - 1; i >= 0; i--) {
      const p = particles[i]
      p.age += elapsed
      const edge = Math.min(p.x + 0.8, width + 0.8 - p.x, p.y + 0.8, height + 0.8 - p.y)
      if (p.age >= p.lifetime || edge <= 0) {
        p.node.remove()
        particles.splice(i, 1)
        continue
      }
      const fade = Math.max(0, Math.min(1, p.age / 1.8, (p.lifetime - p.age) / 2, edge))
      p.node.style.opacity = String(fade * fade * (3 - 2 * fade))
      p.node.setAttribute("transform", `translate(${p.x * 50} ${p.y * 50})`)
    }
    frame = requestAnimationFrame(draw)
  }
  frame = requestAnimationFrame(draw)
  window.addCleanup(() => {
    cancelAnimationFrame(frame)
    observer.disconnect()
    for (const p of particles) p.node.remove()
  })
})
