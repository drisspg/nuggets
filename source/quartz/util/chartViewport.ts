export type ChartRange = [number, number]

// Bound projection magnitudes and avoid zooming beyond useful floating-point resolution.
const MAX_ZOOM = 1_000_000

function fit(start: number, span: number, full: ChartRange): ChartRange | null {
  const width = Math.min(full[1] - full[0], Math.max(span, (full[1] - full[0]) / MAX_ZOOM))
  if (width >= (full[1] - full[0]) * (1 - 8 * Number.EPSILON)) return [...full]
  const low = Math.max(full[0], Math.min(full[1] - width, start))
  const high = Math.min(full[1], low + width)
  return low < high ? [low, high] : null
}

/** Select a normalized interval; reversed drags are equivalent and empty drags are ignored. */
export function selectChartRange(
  view: ChartRange,
  full: ChartRange,
  a: number,
  b: number,
): ChartRange | null {
  if (!Number.isFinite(a) || !Number.isFinite(b)) return null
  const lo = Math.max(0, Math.min(1, Math.min(a, b)))
  const hi = Math.max(0, Math.min(1, Math.max(a, b)))
  if (lo === hi) return null
  const span = view[1] - view[0]
  return fit(view[0] + span * lo, span * (hi - lo), full)
}

/** Pan by a fraction of the current span, preserving scale and staying inside the full extent. */
export function panChartRange(view: ChartRange, full: ChartRange, fraction: number): ChartRange {
  if (!Number.isFinite(fraction)) return [...view]
  const span = view[1] - view[0]
  return fit(view[0] + span * fraction, span, full) ?? [...view]
}

/** Keyboard zoom centered on the viewport, or on a normalized anchor within it. */
export function scaleChartRange(
  view: ChartRange,
  full: ChartRange,
  factor: number,
  anchor = 0.5,
): ChartRange {
  if (!Number.isFinite(factor) || factor <= 0 || !Number.isFinite(anchor)) return [...view]
  anchor = Math.max(0, Math.min(1, anchor))
  const oldSpan = view[1] - view[0]
  const span = Math.min(
    full[1] - full[0],
    Math.max(oldSpan * factor, (full[1] - full[0]) / MAX_ZOOM),
  )
  return fit(view[0] + oldSpan * anchor - span * anchor, span, full) ?? [...view]
}
