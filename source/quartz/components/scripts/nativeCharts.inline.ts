import { axisBottom, axisLeft, format, line, pointer, scaleLinear, select } from "d3"
import { ChartPoint, ChartSpec, numericDomain, parseChart } from "../../util/chart"

const palette = ["blue", "amber", "green", "red", "gold", "purple"] as const
const tickNumber = format(",~g")
const tickScientific = format(".1e")

function element<K extends keyof HTMLElementTagNameMap>(
  tag: K,
  className: string,
  text?: string,
): HTMLElementTagNameMap[K] {
  const node = document.createElement(tag)
  node.className = className
  if (text !== undefined) node.textContent = text
  return node
}

function valueText(value: number | string | null): string {
  return value === null ? "Missing" : String(value)
}

function pointDetails(point: ChartPoint): string {
  return Object.entries(point.details ?? {})
    .map(([label, value]) => `${label}: ${value}`)
    .join(" · ")
}

/** The table retains all observations, including gaps and hidden series. */
function dataTable(spec: ChartSpec): HTMLDetailsElement {
  const details = element("details", "native-chart-data")
  details.append(element("summary", "", "View data table"))
  const scroll = element("div", "native-chart-table-wrap")
  scroll.tabIndex = 0
  scroll.setAttribute("role", "region")
  scroll.setAttribute("aria-label", "Chart data table")
  const table = element("table", "")
  const hasXBounds = spec.series.some((series) => series.points.some((p) => p.xLow !== undefined))
  const hasYBounds = spec.series.some((series) => series.points.some((p) => p.yLow !== undefined))
  const caption = element("caption", "", spec.intervalLabel ?? "Original chart observations")
  table.append(caption)
  const header = table.createTHead().insertRow()
  const labels = ["Series", spec.x.label, spec.y.label]
  if (hasXBounds) labels.push("X lower", "X upper")
  if (hasYBounds) labels.push("Y lower", "Y upper")
  labels.push("Details")
  for (const label of labels) {
    const cell = document.createElement("th")
    cell.scope = "col"
    cell.textContent = label
    header.append(cell)
  }
  const body = table.createTBody()
  for (const series of spec.series) {
    for (const point of series.points) {
      const row = body.insertRow()
      const cells = [series.name, valueText(point.x), valueText(point.y)]
      if (hasXBounds) cells.push(valueText(point.xLow ?? null), valueText(point.xHigh ?? null))
      if (hasYBounds) cells.push(valueText(point.yLow ?? null), valueText(point.yHigh ?? null))
      cells.push(pointDetails(point))
      for (const text of cells) row.insertCell().textContent = text
    }
  }
  scroll.append(table)
  details.append(scroll)
  return details
}

function drawChart(
  figure: HTMLElement,
  spec: ChartSpec,
  signal: AbortSignal,
  cleanups: Array<() => void>,
) {
  const content = figure.querySelector<HTMLElement>(".native-chart-content")!
  content.replaceChildren()
  const title = figure.querySelector("figcaption")!.textContent!
  const height = Number(figure.dataset.chartHeight)
  const categories = "categories" in spec.y ? spec.y.categories : null
  const hidden = new Set<number>()
  const seriesColor = (index: number) => spec.series[index].color ?? palette[index % palette.length]
  const yValue = (point: ChartPoint) =>
    categories ? categories.indexOf(point.y as string) : (point.y as number)
  const observations = spec.series.flatMap((series, index) =>
    series.points
      .filter((point) => point.y !== null)
      .map((point) => ({ series: index, point, key: categories ? yValue(point) : point.x })),
  )
  const legend = element("div", "native-chart-legend")
  legend.setAttribute("aria-label", "Visible chart series")
  const buttons = spec.series.map((series, index) => {
    const button = element("button", "native-chart-legend-item")
    button.type = "button"
    button.style.setProperty("--series-color", `var(--chart-${seriesColor(index)})`)
    const swatch = element("span", "native-chart-swatch")
    swatch.dataset.dash = series.dash ?? "solid"
    swatch.dataset.mode = series.mode
    swatch.dataset.marker = series.marker ?? "circle"
    swatch.setAttribute("aria-hidden", "true")
    button.append(swatch, element("span", "", series.name))
    button.addEventListener(
      "click",
      () => {
        if (hidden.has(index)) hidden.delete(index)
        else if (hidden.size < spec.series.length - 1) hidden.add(index)
        render()
      },
      { signal },
    )
    legend.append(button)
    return button
  })
  const plot = element("div", "native-chart-plot")
  const svgNode = document.createElementNS("http://www.w3.org/2000/svg", "svg")
  svgNode.setAttribute("role", "img")
  svgNode.setAttribute("tabindex", "0")
  svgNode.setAttribute("aria-label", title)
  svgNode.setAttribute(
    "aria-description",
    "Hover or tap to inspect observations. Arrow keys move between observations; Home and End jump to the first and last. Escape clears inspection. The data table below contains all values.",
  )
  const svg = select(svgNode)
  plot.append(svgNode)
  const readout = element("div", "native-chart-readout")
  readout.setAttribute("role", "status")
  readout.setAttribute("aria-live", "off")
  if (spec.series.length > 1) content.append(legend)
  content.append(
    element("div", "native-chart-axis-label native-chart-y-label", spec.y.label),
    plot,
    element("div", "native-chart-axis-label native-chart-x-label", spec.x.label),
    readout,
  )
  if (spec.intervalLabel)
    content.append(element("p", "native-chart-interval-label", spec.intervalLabel))
  content.append(dataTable(spec))

  let width = 0
  let activeKey: number | null = null
  const prompt = "Hover, tap, or focus the plot and use arrow keys to inspect values."

  function render() {
    if (!width) return
    const left = categories
      ? Math.min(
          width * 0.43,
          Math.max(84, Math.max(...categories.map((c) => c.length)) * 6.5 + 14),
        )
      : 64
    const right = Math.max(left + 1, width - 18)
    const top = 10
    const bottom = height - 30
    const x = scaleLinear().domain(numericDomain(spec, "x")).range([left, right])
    const y = scaleLinear()
      .domain(categories ? [-0.5, categories.length - 0.5] : numericDomain(spec, "y"))
      .range(categories ? [top, bottom] : [bottom, top])
    const visible = observations.filter((item) => !hidden.has(item.series))
    const keys = [...new Set(visible.map((item) => item.key))].sort((a, b) => a - b)
    if (activeKey !== null && !keys.includes(activeKey)) activeKey = null
    buttons.forEach((button, index) => {
      button.setAttribute("aria-pressed", String(!hidden.has(index)))
      button.disabled = !hidden.has(index) && hidden.size === spec.series.length - 1
    })
    svg.attr("viewBox", `0 0 ${width} ${height}`).attr("height", height)
    svg.selectAll("*").remove()
    svg.append("title").text(title)
    const xTicks = Math.max(3, Math.floor((right - left) / 95))
    const yTicks = Math.max(2, Math.floor((bottom - top) / 55))
    const xFormat = spec.x.format === "scientific" ? tickScientific : tickNumber
    const yFormat =
      !categories && "format" in spec.y && spec.y.format === "scientific"
        ? tickScientific
        : tickNumber
    if (!categories) {
      svg
        .append("g")
        .attr("class", "native-chart-grid")
        .attr("transform", `translate(${left},0)`)
        .call(
          axisLeft(y)
            .ticks(yTicks)
            .tickSize(-(right - left))
            .tickFormat(() => ""),
        )
        .call((group) => group.select(".domain").remove())
    }
    svg
      .append("g")
      .attr("class", "native-chart-axis")
      .attr("transform", `translate(0,${bottom})`)
      .call(
        axisBottom(x)
          .ticks(xTicks)
          .tickSizeOuter(0)
          .tickFormat((value) => xFormat(Number(value))),
      )
    const yAxis = axisLeft(y).tickSizeOuter(0)
    if (categories) {
      const maxCharacters = Math.max(6, Math.floor((left - 12) / 6.5))
      yAxis.tickValues(categories.map((_, i) => i)).tickFormat((value) => {
        const label = categories[Number(value)]
        return label.length > maxCharacters ? `${label.slice(0, maxCharacters - 1)}…` : label
      })
    } else yAxis.ticks(yTicks).tickFormat((value) => yFormat(Number(value)))
    const yAxisGroup = svg
      .append("g")
      .attr("class", "native-chart-axis")
      .attr("transform", `translate(${left},0)`)
      .call(yAxis)
    if (categories)
      yAxisGroup
        .selectAll<SVGTextElement, number>(".tick text")
        .append("title")
        .text((value) => categories[value])
    if (spec.x.includeZero) {
      svg
        .append("line")
        .attr("class", "native-chart-zero")
        .attr("x1", x(0))
        .attr("x2", x(0))
        .attr("y1", top)
        .attr("y2", bottom)
    }
    if (!categories && "includeZero" in spec.y && spec.y.includeZero) {
      svg
        .append("line")
        .attr("class", "native-chart-zero")
        .attr("x1", left)
        .attr("x2", right)
        .attr("y1", y(0))
        .attr("y2", y(0))
    }
    spec.series.forEach((series, index) => {
      if (hidden.has(index)) return
      const group = svg
        .append("g")
        .attr("class", "native-chart-series")
        .attr("data-series", index)
        .style("color", `var(--chart-${seriesColor(index)})`)
      if (series.mode === "line") {
        group
          .append("path")
          .attr("class", "native-chart-line")
          .attr("stroke-dasharray", series.dash === "dash" ? "5 4" : null)
          .attr(
            "d",
            line<ChartPoint>()
              .defined((p) => p.y !== null)
              .x((p) => x(p.x))
              .y((p) => y(yValue(p)))(series.points),
          )
      }
      for (const point of series.points) {
        if (point.y === null) continue
        const px = x(point.x)
        const py = y(yValue(point))
        if (point.xLow !== undefined) {
          const low = x(point.xLow),
            high = x(point.xHigh!)
          group
            .append("path")
            .attr("class", "native-chart-error native-chart-error-x")
            .attr("d", `M${low},${py}H${high}M${low},${py - 3}v6M${high},${py - 3}v6`)
        }
        if (point.yLow !== undefined) {
          const low = y(point.yLow),
            high = y(point.yHigh!)
          group
            .append("path")
            .attr("class", "native-chart-error native-chart-error-y")
            .attr("d", `M${px},${low}V${high}M${px - 3},${low}h6M${px - 3},${high}h6`)
        }
        group
          .append("path")
          .attr("class", "native-chart-point")
          .attr("data-x", String(point.x))
          .attr("data-y", String(point.y))
          .attr("transform", `translate(${px},${py})`)
          .attr(
            "d",
            series.marker === "diamond"
              ? "M0,-4.5L4.5,0L0,4.5L-4.5,0Z"
              : "M-3,0a3,3 0 1,0 6,0a3,3 0 1,0 -6,0",
          )
      }
    })
    const inspection = svg
      .append("g")
      .attr("class", "native-chart-inspection")
      .attr("pointer-events", "none")

    function inspect(key: number | null, announce = false) {
      activeKey = key
      inspection.selectAll("*").remove()
      readout.setAttribute("aria-live", announce ? "polite" : "off")
      readout.replaceChildren()
      if (key === null) {
        readout.append(element("p", "native-chart-hint", prompt))
        return
      }
      const matches = visible.filter((item) => item.key === key)
      const label = categories ? categories[key] : `${spec.x.label}: ${valueText(key)}`
      readout.append(element("div", "native-chart-readout-coordinate", label))
      const values = element("div", "native-chart-readout-values")
      if (categories) {
        inspection
          .append("line")
          .attr("class", "native-chart-crosshair")
          .attr("x1", left)
          .attr("x2", right)
          .attr("y1", y(key))
          .attr("y2", y(key))
      } else {
        inspection
          .append("line")
          .attr("class", "native-chart-crosshair")
          .attr("x1", x(key))
          .attr("x2", x(key))
          .attr("y1", top)
          .attr("y2", bottom)
      }
      for (const { point, series } of matches) {
        const row = element("div", "native-chart-readout-value")
        row.style.setProperty("--series-color", `var(--chart-${seriesColor(series)})`)
        row.append(element("strong", "", spec.series[series].name))
        row.append(
          element(
            "span",
            "",
            `${categories ? spec.x.label : spec.y.label}: ${valueText(categories ? point.x : point.y)}`,
          ),
        )
        if (point.xLow !== undefined)
          row.append(element("span", "", `X interval: [${point.xLow}, ${point.xHigh}]`))
        if (point.yLow !== undefined)
          row.append(element("span", "", `Y interval: [${point.yLow}, ${point.yHigh}]`))
        const metadata = pointDetails(point)
        if (metadata) row.append(element("span", "", metadata))
        values.append(row)
        inspection
          .append("circle")
          .attr("class", "native-chart-active-point")
          .attr("cx", x(point.x))
          .attr("cy", y(yValue(point)))
          .attr("r", 5)
          .style("stroke", `var(--chart-${seriesColor(series)})`)
      }
      readout.append(values)
    }

    function inspectPointer(event: PointerEvent) {
      const [px, py] = pointer(event, svgNode)
      if (px < left || px > right || py < top || py > bottom) return
      const position = categories ? y.invert(py) : x.invert(px)
      const nearest = keys.reduce((best, key) =>
        Math.abs(key - position) < Math.abs(best - position) ? key : best,
      )
      if (nearest !== activeKey) inspect(nearest)
    }
    svg
      .on("pointermove.nativeChart", inspectPointer)
      .on("pointerdown.nativeChart", inspectPointer)
      .on("focus.nativeChart", () => {
        if (activeKey === null) inspect(keys[0], true)
      })
      .on("keydown.nativeChart", (event: KeyboardEvent) => {
        if (event.ctrlKey || event.metaKey || event.altKey) return
        let index = activeKey === null ? -1 : keys.indexOf(activeKey)
        switch (event.key) {
          case "ArrowRight":
          case "ArrowDown":
            index = Math.min(keys.length - 1, index + 1)
            break
          case "ArrowLeft":
          case "ArrowUp":
            index = Math.max(0, index - 1)
            break
          case "Home":
            index = 0
            break
          case "End":
            index = keys.length - 1
            break
          case "Escape":
            event.preventDefault()
            inspect(null, true)
            return
          default:
            return
        }
        event.preventDefault()
        inspect(keys[index], true)
      })
    inspect(activeKey)
  }

  const resize = new ResizeObserver(([entry]) => {
    const next = Math.floor(entry.contentRect.width)
    if (next > 0 && next !== width) {
      width = next
      render()
    }
  })
  resize.observe(plot)
  cleanups.push(() => {
    resize.disconnect()
    svg.on(".nativeChart", null)
  })
}

document.addEventListener("nav", () => {
  const controller = new AbortController()
  const cleanups: Array<() => void> = []
  window.addCleanup(() => {
    controller.abort()
    cleanups.forEach((cleanup) => cleanup())
  })
  for (const figure of document.querySelectorAll<HTMLElement>(".native-chart")) {
    const content = figure.querySelector<HTMLElement>(".native-chart-content")!
    content.replaceChildren(element("p", "native-chart-hint", "Loading chart…"))
    void (async () => {
      try {
        const response = await fetch(figure.dataset.chartSrc!, { signal: controller.signal })
        if (!response.ok) throw new Error(`Chart request failed (${response.status})`)
        const spec = parseChart(await response.json())
        if (!controller.signal.aborted && figure.isConnected) {
          drawChart(figure, spec, controller.signal, cleanups)
        }
      } catch (error) {
        if (controller.signal.aborted) return
        const message = element(
          "p",
          "native-chart-error-message",
          "Chart unavailable. You can still download its data below.",
        )
        message.setAttribute("role", "alert")
        content.replaceChildren(message)
        console.warn("Unable to load native chart", figure.dataset.chartSrc, error)
      }
    })()
  }
})
