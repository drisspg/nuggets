export type NumericAxis = {
  label: string
  format?: "number" | "scientific"
  includeZero?: boolean
  domain?: [number, number]
}

export type CategoryAxis = { label: string; categories: string[] }

export type ChartPoint = {
  x: number
  y: number | string | null
  xLow?: number
  xHigh?: number
  yLow?: number
  yHigh?: number
  details?: Record<string, string | number>
}

export type ChartSeries = {
  name: string
  mode: "line" | "points"
  color?: "blue" | "amber" | "green" | "red" | "gold" | "purple"
  dash?: "solid" | "dash"
  marker?: "circle" | "diamond"
  points: ChartPoint[]
}

export type ChartSpec = {
  version: 1
  x: NumericAxis
  y: NumericAxis | CategoryAxis
  intervalLabel?: string
  series: ChartSeries[]
}

function fail(location: string, message: string): never {
  throw new Error(`${location}: ${message}`)
}

function record(value: unknown, location: string, keys?: string[]): Record<string, unknown> {
  if (
    value === null ||
    typeof value !== "object" ||
    ![Object.prototype, null].includes(Object.getPrototypeOf(value))
  ) {
    fail(location, "expected a plain object")
  }
  if (keys) {
    for (const key of Object.keys(value)) {
      if (!keys.includes(key)) fail(`${location}.${key}`, "unknown field")
    }
  }
  return value as Record<string, unknown>
}

function text(value: unknown, location: string): string {
  if (typeof value !== "string" || !value.trim()) fail(location, "expected a nonempty string")
  return value
}

function finite(value: unknown, location: string): number {
  if (typeof value !== "number" || !Number.isFinite(value)) {
    fail(location, "expected a finite number")
  }
  return value
}

function choice<const T extends readonly string[]>(
  value: unknown,
  options: T,
  location: string,
): T[number] {
  if (typeof value !== "string" || !options.includes(value)) {
    fail(location, `expected one of: ${options.join(", ")}`)
  }
  return value as T[number]
}

function nonemptyArray(value: unknown, location: string): unknown[] {
  if (!Array.isArray(value) || value.length === 0) fail(location, "expected a nonempty array")
  return Array.from(value)
}

function numericAxis(value: unknown, location: string): NumericAxis {
  const input = record(value, location, ["label", "format", "includeZero", "domain"])
  const axis: NumericAxis = { label: text(input.label, `${location}.label`) }
  if ("format" in input)
    axis.format = choice(input.format, ["number", "scientific"], `${location}.format`)
  if ("includeZero" in input) {
    if (typeof input.includeZero !== "boolean")
      fail(`${location}.includeZero`, "expected a boolean")
    axis.includeZero = input.includeZero
  }
  if ("domain" in input) {
    if (!Array.isArray(input.domain) || input.domain.length !== 2) {
      fail(`${location}.domain`, "expected a pair of finite numbers")
    }
    const low = finite(input.domain[0], `${location}.domain[0]`)
    const high = finite(input.domain[1], `${location}.domain[1]`)
    axis.domain = [low, high]
  }
  return axis
}

/** Validate v1 chart data at both the build and fetch boundaries, without mutating the input. */
export function parseChart(value: unknown): ChartSpec {
  const input = record(value, "chart", ["version", "x", "y", "intervalLabel", "series"])
  if (input.version !== 1) fail("chart.version", "expected version 1")
  const x = numericAxis(input.x, "chart.x")
  const yInput = record(input.y, "chart.y")
  let y: NumericAxis | CategoryAxis
  let categories: Set<string> | undefined
  if ("categories" in yInput) {
    record(yInput, "chart.y", ["label", "categories"])
    categories = new Set<string>()
    y = {
      label: text(yInput.label, "chart.y.label"),
      categories: nonemptyArray(yInput.categories, "chart.y.categories").map((value, index) => {
        const location = `chart.y.categories[${index}]`
        const category = text(value, location)
        if (categories!.has(category))
          fail(location, `duplicate category ${JSON.stringify(category)}`)
        categories!.add(category)
        return category
      }),
    }
  } else {
    y = numericAxis(yInput, "chart.y")
  }
  const chart: ChartSpec = { version: 1, x, y, series: [] }
  if ("intervalLabel" in input)
    chart.intervalLabel = text(input.intervalLabel, "chart.intervalLabel")
  const names = new Set<string>()
  chart.series = nonemptyArray(input.series, "chart.series").map((value, seriesIndex) => {
    const location = `chart.series[${seriesIndex}]`
    const input = record(value, location, ["name", "mode", "color", "dash", "marker", "points"])
    const name = text(input.name, `${location}.name`)
    if (names.has(name)) fail(`${location}.name`, `duplicate series name ${JSON.stringify(name)}`)
    names.add(name)
    const mode = choice(input.mode, ["line", "points"], `${location}.mode`)
    if (categories && mode !== "points")
      fail(`${location}.mode`, "category charts require points mode")
    const series: ChartSeries = { name, mode, points: [] }
    if ("color" in input) {
      series.color = choice(
        input.color,
        ["blue", "amber", "green", "red", "gold", "purple"],
        `${location}.color`,
      )
    }
    if ("dash" in input) series.dash = choice(input.dash, ["solid", "dash"], `${location}.dash`)
    if ("marker" in input)
      series.marker = choice(input.marker, ["circle", "diamond"], `${location}.marker`)
    let previousX = -Infinity
    let plottable = false
    series.points = nonemptyArray(input.points, `${location}.points`).map((value, pointIndex) => {
      const pointLocation = `${location}.points[${pointIndex}]`
      const input = record(value, pointLocation, [
        "x",
        "y",
        "xLow",
        "xHigh",
        "yLow",
        "yHigh",
        "details",
      ])
      const x = finite(input.x, `${pointLocation}.x`)
      if (mode === "line" && x <= previousX) {
        fail(`${pointLocation}.x`, "line points must have strictly increasing x values")
      }
      previousX = x
      let y: ChartPoint["y"]
      if (categories) {
        y = text(input.y, `${pointLocation}.y`)
        if (!categories.has(y)) fail(`${pointLocation}.y`, `unknown category ${JSON.stringify(y)}`)
      } else {
        y = input.y === null ? null : finite(input.y, `${pointLocation}.y`)
      }
      if (y !== null) plottable = true
      const point: ChartPoint = { x, y }
      for (const axis of ["x", "y"] as const) {
        const lowKey = `${axis}Low` as const
        const highKey = `${axis}High` as const
        if (!(lowKey in input) && !(highKey in input)) continue
        if (!(lowKey in input) || !(highKey in input)) {
          fail(pointLocation, `${lowKey} and ${highKey} must be provided together`)
        }
        const estimate = point[axis]
        if (y === null || typeof estimate !== "number") {
          fail(
            `${pointLocation}.${lowKey}`,
            "bounds require a numeric estimate (not a gap or category)",
          )
        }
        const low = finite(input[lowKey], `${pointLocation}.${lowKey}`)
        const high = finite(input[highKey], `${pointLocation}.${highKey}`)
        if (low > high || low > estimate || high < estimate) {
          fail(`${pointLocation}.${lowKey}`, "bounds must be ordered and enclose the estimate")
        }
        if (!chart.intervalLabel)
          fail("chart.intervalLabel", `required for intervals at ${pointLocation}`)
        point[lowKey] = low
        point[highKey] = high
      }
      if ("details" in input) {
        const details = record(input.details, `${pointLocation}.details`)
        point.details = Object.fromEntries(
          Object.entries(details).map(([key, value]) => {
            if (typeof value !== "string") finite(value, `${pointLocation}.details.${key}`)
            return [key, value as string | number]
          }),
        )
      }
      return point
    })
    if (!plottable) fail(`${location}.points`, "series must contain a plottable point")
    return series
  })
  numericDomain(chart, "x")
  if (!categories) numericDomain(chart, "y")
  return chart
}

/** Domains always include every series, including hidden series and interval endpoints. */
export function numericDomain(chart: ChartSpec, axis: "x" | "y"): [number, number] {
  const config = chart[axis]
  const location = `chart.${axis}.domain`
  if ("categories" in config) fail(location, "category axes do not have a numeric domain")
  let low = Infinity
  let high = -Infinity
  for (const series of chart.series) {
    for (const point of series.points) {
      const estimate = point[axis]
      if (typeof estimate !== "number") continue
      low = Math.min(low, point[`${axis}Low`] ?? estimate)
      high = Math.max(high, point[`${axis}High`] ?? estimate)
    }
  }
  if (config.includeZero) {
    low = Math.min(low, 0)
    high = Math.max(high, 0)
  }
  if (config.domain) {
    const min = finite(config.domain[0], `${location}[0]`)
    const max = finite(config.domain[1], `${location}[1]`)
    if (min >= max) fail(location, "must be strictly increasing")
    if (!Number.isFinite(max - min)) fail(location, "range overflows")
    if (min > low || max < high)
      fail(location, "must encompass all data, intervals, and requested zero")
    return [min, max]
  }
  const range = high - low
  if (!Number.isFinite(range)) fail(location, "range overflows")
  const padding = Math.max((range || Math.abs(low) || 1) * 0.05, Number.MIN_VALUE)
  const min = low - padding
  const max = high + padding
  if (!Number.isFinite(min) || !Number.isFinite(max) || !Number.isFinite(max - min)) {
    fail(location, "padded range overflows")
  }
  return [min, max]
}
