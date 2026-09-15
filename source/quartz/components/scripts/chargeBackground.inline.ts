import { chargeMotion } from "../../util/chargeMotion"

document.addEventListener("nav", () => {
  const svg = document.querySelector<SVGSVGElement>(".home-collision")
  const left = svg?.querySelector<SVGGElement>(".charge-packet-a")
  const right = svg?.querySelector<SVGGElement>(".charge-packet-b")
  if (!svg || !left || !right) return

  const start = performance.now() - 2000
  let frame: number
  const draw = (now: number) => {
    const phase = ((now - start) % 18000) / 18000
    const { separation } = chargeMotion((phase - 0.5) * 8)
    const half = separation * 25
    left.setAttribute("transform", `translate(${390 - half} 115)`)
    right.setAttribute("transform", `translate(${390 + half} 115)`)
    // Fade only the replay boundary, not the physical turnaround at closest approach.
    const fade = Math.min(1, phase / 0.08, (1 - phase) / 0.08)
    svg.style.opacity = String(0.6 * fade * fade * (3 - 2 * fade))
    frame = requestAnimationFrame(draw)
  }
  frame = requestAnimationFrame(draw)
  window.addCleanup(() => cancelAnimationFrame(frame))
})
