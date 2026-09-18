import { axisBottom, axisLeft, format, line, pointer, scaleLinear, select } from "d3"
import { ChartPoint, ChartSpec, numericDomain, parseChart } from "../../util/chart"
import {
  panChartRange,
  scaleChartRange,
  selectChartRange,
  type ChartRange,
} from "../../util/chartViewport"

const palette = ["blue", "amber", "green", "red", "gold", "purple"] as const
const tickNumber = format(",~g")
const tickScientific = format(".1e")
let nextClipId = 0

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

function toolButton(label: string, icon: string): HTMLButtonElement {
  const button = element("button", "native-chart-tool")
  button.type = "button"
  button.title = label
  button.setAttribute("aria-label", label)
  // Icons are static markup owned by this module, never chart data.
  button.innerHTML = `<svg viewBox="0 0 24 24" aria-hidden="true">${icon}</svg>`
  return button
}

function valueText(value: number | string | null): string {
  return value === null ? "Missing" : String(value)
}

function pointDetails(point: ChartPoint): string {
  return Object.entries(point.details ?? {})
    .map(([label, value]) => `${label}: ${value}`)
    .join(" · ")
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
  const controls = spec.series.map((series, index) => {
    const button = element("button", "native-chart-legend-item")
    button.type = "button"
    button.style.setProperty("--series-color", `var(--chart-${seriesColor(index)})`)
    const swatch = element("span", "native-chart-swatch")
    swatch.dataset.dash = series.dash ?? "solid"
    swatch.dataset.mode = series.mode
    swatch.dataset.marker = series.marker ?? "circle"
    swatch.setAttribute("aria-hidden", "true")
    const value = element("span", "native-chart-legend-value", "—")
    button.append(swatch, element("span", "native-chart-legend-name", series.name), value)
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
    return { button, value }
  })
  const plot = element("div", "native-chart-plot")
  const svgNode = document.createElementNS("http://www.w3.org/2000/svg", "svg")
  svgNode.setAttribute("role", "group")
  svgNode.setAttribute("aria-roledescription", "interactive chart")
  svgNode.setAttribute("tabindex", "0")
  svgNode.setAttribute("aria-label", title)
  svgNode.setAttribute(
    "aria-description",
    "Hover to inspect. Click or tap to pin; click again to release. Drag a region to zoom; Shift-drag to pan. On touch, activate the zoom tool before dragging. Plus and minus zoom, Shift-arrow keys pan, and zero resets the view. Arrow keys inspect points; Enter or Space toggles pinning. Escape cancels a drag or exits interaction without scrolling. Full-precision values remain in legend tooltips and keyboard inspection.",
  )
  const svg = select(svgNode)
  plot.append(svgNode)
  const tools = element("div", "native-chart-tools")
  const pinButton = toolButton(
    "Pin inspected point",
    '<g class="native-chart-cursor-icon"><circle cx="12" cy="12" r="4"/><path d="M12 2v4m0 12v4M2 12h4m12 0h4"/></g><g class="native-chart-pin-icon"><path d="M8 3h8m-7 0v5l-3 5v2h12v-2l-3-5V3m-3 12v7"/></g>',
  )
  pinButton.classList.add("native-chart-pin-toggle")
  const zoomButton = toolButton(
    "Drag to zoom; Shift-drag to pan. On touch, activate this tool first.",
    '<circle cx="10" cy="10" r="6"/><path d="m15 15 6 6M7 10h6m-3-3v6"/>',
  )
  zoomButton.classList.add("native-chart-zoom-toggle")
  const resetButton = toolButton("Reset zoom (0)", '<path d="M4 10a8 8 0 1 1 1 7M4 4v6h6"/>')
  resetButton.classList.add("native-chart-reset")
  tools.append(pinButton, zoomButton, resetButton)
  const coordinate = element("div", "native-chart-coordinate")
  const coordinateValue = element("span", "native-chart-coordinate-value")
  coordinate.append(coordinateValue)
  const reduceMotion = matchMedia("(prefers-reduced-motion: reduce)")
  const announcement = element("span", "native-chart-status")
  announcement.setAttribute("role", "status")
  content.append(
    tools,
    legend,
    coordinate,
    element("div", "native-chart-axis-label native-chart-y-label", spec.y.label),
    plot,
    element("div", "native-chart-axis-label native-chart-x-label", spec.x.label),
    announcement,
  )
  if (spec.intervalLabel)
    content.append(element("p", "native-chart-interval-label", spec.intervalLabel))

  let width = 0
  let activeKey: number | null = null
  let pinned = false
  const full = {
    x: numericDomain(spec, "x"),
    y: categories ? ([-0.5, categories.length - 0.5] as ChartRange) : numericDomain(spec, "y"),
  }
  let view = full
  let zoomArmed = false
  let suppressClick = false
  type Drag = {
    id: number
    start: [number, number]
    view: typeof full
    pan: boolean
    moved: boolean
    threshold: number
    bounds: [number, number, number, number]
  }
  let drag: Drag | null = null
  const clipId = `native-chart-clip-${++nextClipId}`
  const prompt = "Hover to inspect · drag to zoom"
  const valueFormat = categories ? spec.x.format : "format" in spec.y ? spec.y.format : undefined
  const legendNumber = format(valueFormat === "scientific" ? ".6e" : ".7~g")

  function endCapture(cancel = false) {
    const id = drag?.id
    if (drag) {
      if (cancel) view = drag.view
      if (drag.moved || cancel) suppressClick = true
    }
    drag = null
    if (id !== undefined && svgNode.hasPointerCapture(id)) svgNode.releasePointerCapture(id)
  }

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
    const x = scaleLinear().domain(view.x).range([left, right])
    const y = scaleLinear()
      .domain(view.y)
      .range(categories ? [top, bottom] : [bottom, top])
    const visible = observations.filter((item) => !hidden.has(item.series))
    const inView = (point: ChartPoint) =>
      point.x >= view.x[0] &&
      point.x <= view.x[1] &&
      (point.y === null ||
        categories !== null ||
        (yValue(point) >= view.y[0] && yValue(point) <= view.y[1]))
    const keys = [
      ...new Set(
        spec.series.flatMap((series, index) =>
          hidden.has(index)
            ? []
            : series.points.filter(inView).map((point) => (categories ? yValue(point) : point.x)),
        ),
      ),
    ].sort((a, b) => a - b)
    if (activeKey !== null && !keys.includes(activeKey)) {
      activeKey = null
      pinned = false
    }
    controls.forEach(({ button }, index) => {
      button.setAttribute("aria-pressed", String(!hidden.has(index)))
      button.setAttribute(
        "aria-disabled",
        String(!hidden.has(index) && hidden.size === spec.series.length - 1),
      )
    })
    const zoomed =
      view.x.some((value, i) => value !== full.x[i]) ||
      view.y.some((value, i) => value !== full.y[i])
    figure.dataset.zoomed = String(zoomed)
    figure.dataset.zoomArmed = String(zoomArmed)
    figure.dataset.dragging = drag?.moved ? (drag.pan ? "pan" : "zoom") : ""
    zoomButton.setAttribute("aria-pressed", String(zoomArmed))
    resetButton.hidden = !zoomed
    svgNode.style.touchAction = zoomArmed ? "none" : "pan-y"
    svg
      .attr("viewBox", `0 0 ${width} ${height}`)
      .attr("height", height)
      .attr("data-x-domain", JSON.stringify(view.x))
      .attr("data-y-domain", JSON.stringify(view.y))
    svg.selectAll("*").remove()
    svg.append("title").text(title)
    svg
      .append("defs")
      .append("clipPath")
      .attr("id", clipId)
      .append("rect")
      .attr("x", left)
      .attr("y", top)
      .attr("width", right - left)
      .attr("height", bottom - top)
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
    if (spec.x.includeZero && view.x[0] <= 0 && view.x[1] >= 0) {
      svg
        .append("line")
        .attr("class", "native-chart-zero")
        .attr("x1", x(0))
        .attr("x2", x(0))
        .attr("y1", top)
        .attr("y2", bottom)
    }
    if (
      !categories &&
      "includeZero" in spec.y &&
      spec.y.includeZero &&
      view.y[0] <= 0 &&
      view.y[1] >= 0
    ) {
      svg
        .append("line")
        .attr("class", "native-chart-zero")
        .attr("x1", left)
        .attr("x2", right)
        .attr("y1", y(0))
        .attr("y2", y(0))
    }
    const dataLayer = svg
      .append("g")
      .attr("class", "native-chart-data-layer")
      .attr("clip-path", `url(#${clipId})`)
    spec.series.forEach((series, index) => {
      if (hidden.has(index)) return
      const group = dataLayer
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
      // Keep every line vertex and inspection point, but don't stack dense marker glyphs.
      const markerDiameter = series.marker === "diamond" ? 9 : 6
      const denseLine =
        series.mode === "line" &&
        series.points.filter((point) => point.y !== null).length * markerDiameter > right - left
      const showMarkers =
        !denseLine ||
        series.points.filter((point) => point.y !== null && inView(point)).length *
          markerDiameter <=
          right - left
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
        if (!showMarkers || (denseLine && !inView(point))) continue
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
    const inspection = dataLayer
      .append("g")
      .attr("class", "native-chart-inspection")
      .attr("pointer-events", "none")

    function inspect(key: number | null, announce = false) {
      activeKey = key
      inspection.selectAll("*").remove()
      const matches = visible.filter((item) => item.key === key)
      const label =
        key === null
          ? keys.length
            ? ""
            : "No point centers in this view · reset zoom"
          : categories
            ? categories[key]
            : `${spec.x.label}: ${valueText(key)}`
      coordinateValue.textContent = label
      coordinateValue.title = label
      figure.dataset.pinned = String(pinned)
      pinButton.setAttribute("aria-pressed", String(pinned))
      pinButton.setAttribute("aria-disabled", String(keys.length === 0))
      pinButton.title = pinned ? "Release pinned point (Escape)" : "Follow cursor · click to pin"
      pinButton.setAttribute("aria-label", pinned ? "Release pinned point" : "Pin inspected point")
      controls.forEach(({ button, value }, index) => {
        const points = matches.filter((item) => item.series === index).map((item) => item.point)
        const missing =
          !hidden.has(index) &&
          !categories &&
          spec.series[index].points.some((point) => point.x === key && point.y === null)
        const nextValue =
          points
            .map((point) => legendNumber((categories ? point.x : point.y) as number))
            .join(", ") || (missing ? "Missing" : "—")
        const changed = nextValue !== value.textContent
        value.textContent = nextValue
        button.dataset.inspecting = String(points.length > 0 || missing)
        if (changed) {
          value.getAnimations().forEach((animation) => animation.cancel())
          if (key !== null && !reduceMotion.matches) {
            value.animate([{ opacity: 0.72 }, { opacity: 1 }], {
              duration: 160,
              easing: "ease-out",
            })
          }
        }
        const description =
          points
            .map((point) => {
              const parts = [
                `${categories ? spec.x.label : spec.y.label}: ${valueText(categories ? point.x : point.y)}`,
              ]
              if (point.xLow !== undefined)
                parts.push(`X interval: [${point.xLow}, ${point.xHigh}]`)
              if (point.yLow !== undefined)
                parts.push(`Y interval: [${point.yLow}, ${point.yHigh}]`)
              const metadata = pointDetails(point)
              if (metadata) parts.push(metadata)
              return parts.join("; ")
            })
            .join("\n") ||
          (hidden.has(index)
            ? "Hidden series"
            : missing
              ? "Missing observation"
              : key === null
                ? prompt
                : "No observation at this position")
        button.title = description
        button.setAttribute("aria-label", `${spec.series[index].name}. ${description}`)
        value.title = description
      })
      if (announce)
        announcement.textContent = [
          `${pinned ? "Pinned" : "Following cursor"}. ${label}`,
          ...controls.map(({ button }) => button.getAttribute("aria-label")),
        ].join(". ")
      if (key === null) return
      if (categories) {
        inspection
          .append("line")
          .attr("class", "native-chart-crosshair")
          .attr("data-axis", "y")
          .attr("x1", left)
          .attr("x2", right)
          .attr("y1", y(key))
          .attr("y2", y(key))
      } else {
        inspection
          .append("line")
          .attr("class", "native-chart-crosshair")
          .attr("data-axis", "x")
          .attr("x1", x(key))
          .attr("x2", x(key))
          .attr("y1", top)
          .attr("y2", bottom)
      }
      // Share one guide per distinct value when several series coincide.
      const otherValues = new Set(
        matches.map(({ point }) => (categories ? point.x : yValue(point))),
      )
      for (const value of otherValues) {
        inspection
          .append("line")
          .attr("class", "native-chart-crosshair")
          .attr("data-axis", categories ? "x" : "y")
          .attr("x1", categories ? x(value) : left)
          .attr("x2", categories ? x(value) : right)
          .attr("y1", categories ? top : y(value))
          .attr("y2", categories ? bottom : y(value))
      }
      for (const { point, series } of matches) {
        inspection
          .append("circle")
          .attr("class", "native-chart-active-point")
          .attr("cx", x(point.x))
          .attr("cy", y(yValue(point)))
          .attr("r", 5)
          .style("stroke", `var(--chart-${seriesColor(series)})`)
      }
    }

    const selection = svg
      .append("rect")
      .attr("class", "native-chart-selection")
      .attr("pointer-events", "none")
      .attr("display", "none")

    function resetView() {
      endCapture(true)
      zoomArmed = false
      view = full
      render()
      announcement.textContent = "Full chart view restored."
    }

    function exitInteraction(event: KeyboardEvent) {
      event.preventDefault()
      endCapture(true)
      zoomArmed = false
      pinned = false
      activeKey = null
      render()
      announcement.textContent = "Chart interaction ended."
      if (
        document.activeElement instanceof HTMLElement ||
        document.activeElement instanceof SVGSVGElement
      )
        document.activeElement.blur()
    }

    select(pinButton).on("click.nativeChart", () => {
      if (!keys.length) return
      pinned = !pinned
      inspect(activeKey ?? keys[0], true)
    })
    select(zoomButton).on("click.nativeChart", () => {
      zoomArmed = !zoomArmed
      render()
      svgNode.focus({ preventScroll: true })
    })
    select(resetButton).on("click.nativeChart", () => {
      resetView()
      svgNode.focus({ preventScroll: true })
    })
    select(content).on("keydown.nativeChart", (event: KeyboardEvent) => {
      if (event.key === "Escape" && !event.defaultPrevented) exitInteraction(event)
    })

    function pointerKey(event: MouseEvent): number | undefined {
      if (!keys.length) return
      const [px, py] = pointer(event, svgNode)
      if (px < left || px > right || py < top || py > bottom) return
      const position = categories ? y.invert(py) : x.invert(px)
      return keys.reduce((best, key) =>
        Math.abs(key - position) < Math.abs(best - position) ? key : best,
      )
    }

    function cancelDrag() {
      if (!drag) return
      endCapture(true)
      zoomArmed = false
      render()
    }

    svg
      .on("pointerdown.nativeChart", (event: PointerEvent) => {
        suppressClick = false
        if (
          event.button !== 0 ||
          !event.isPrimary ||
          drag ||
          (event.pointerType === "touch" && !zoomArmed)
        )
          return
        const [px, py] = pointer(event, svgNode)
        if (px < left || px > right || py < top || py > bottom) return
        drag = {
          id: event.pointerId,
          start: [px, py],
          view,
          pan: event.shiftKey,
          moved: false,
          threshold: event.pointerType === "touch" ? 12 : 6,
          bounds: [left, top, right, bottom],
        }
        svgNode.setPointerCapture(event.pointerId)
        event.preventDefault()
        svgNode.focus({ preventScroll: true })
      })
      .on("pointermove.nativeChart", (event: PointerEvent) => {
        if (drag && event.pointerId === drag.id) {
          const [px, py] = pointer(event, svgNode)
          if (!drag.moved && Math.hypot(px - drag.start[0], py - drag.start[1]) < drag.threshold)
            return
          drag.moved = true
          figure.dataset.dragging = drag.pan ? "pan" : "zoom"
          const [l, t, r, b] = drag.bounds
          if (drag.pan) {
            pinned = false
            activeKey = null
            view = {
              x: panChartRange(drag.view.x, full.x, (drag.start[0] - px) / (r - l)),
              y: categories
                ? full.y
                : panChartRange(drag.view.y, full.y, (py - drag.start[1]) / (b - t)),
            }
            render()
          } else {
            const cx = Math.max(l, Math.min(r, px)),
              cy = Math.max(t, Math.min(b, py))
            const horizontal = Math.abs(cx - drag.start[0]) >= drag.threshold
            const vertical = !categories && Math.abs(cy - drag.start[1]) >= drag.threshold
            if (!horizontal && !vertical) {
              selection.attr("display", "none")
              return
            }
            pinned = false
            inspect(null)
            selection
              .attr("display", null)
              .attr("x", horizontal ? Math.min(cx, drag.start[0]) : l)
              .attr("y", vertical ? Math.min(cy, drag.start[1]) : t)
              .attr("width", horizontal ? Math.abs(cx - drag.start[0]) : r - l)
              .attr("height", vertical ? Math.abs(cy - drag.start[1]) : b - t)
          }
          return
        }
        if (pinned || zoomArmed || event.pointerType === "touch") return
        const key = pointerKey(event)
        if (key !== undefined && key !== activeKey) inspect(key)
      })
      .on("pointerup.nativeChart", (event: PointerEvent) => {
        if (!drag || event.pointerId !== drag.id) return
        const gesture = drag
        const [px, py] = pointer(event, svgNode)
        endCapture()
        if (!gesture.moved) return
        const [l, t, r, b] = gesture.bounds
        const cx = Math.max(l, Math.min(r, px)),
          cy = Math.max(t, Math.min(b, py))
        if (!gesture.pan) {
          const selectedX =
            Math.abs(cx - gesture.start[0]) >= gesture.threshold
              ? selectChartRange(
                  gesture.view.x,
                  full.x,
                  (gesture.start[0] - l) / (r - l),
                  (cx - l) / (r - l),
                )
              : null
          const selectedY =
            !categories && Math.abs(cy - gesture.start[1]) >= gesture.threshold
              ? selectChartRange(
                  gesture.view.y,
                  full.y,
                  (b - gesture.start[1]) / (b - t),
                  (b - cy) / (b - t),
                )
              : null
          view = { x: selectedX ?? gesture.view.x, y: selectedY ?? gesture.view.y }
        }
        zoomArmed = false
        render()
        const changed =
          view.x.some((value, i) => value !== gesture.view.x[i]) ||
          view.y.some((value, i) => value !== gesture.view.y[i])
        announcement.textContent = changed
          ? gesture.pan
            ? "Chart view panned."
            : "Chart view zoomed. Press zero to reset."
          : "View unchanged."
      })
      .on("pointercancel.nativeChart", (event: PointerEvent) => {
        if (drag?.id === event.pointerId) cancelDrag()
      })
      .on("lostpointercapture.nativeChart", (event: PointerEvent) => {
        if (drag?.id === event.pointerId) cancelDrag()
      })
      .on("click.nativeChart", (event: MouseEvent) => {
        if (suppressClick) {
          suppressClick = false
          return
        }
        if (zoomArmed) return
        const key = pointerKey(event)
        if (key === undefined) return
        pinned = !pinned
        inspect(key, true)
        svgNode.focus({ preventScroll: true })
      })
      .on("dblclick.nativeChart", (event: MouseEvent) => {
        event.preventDefault()
        pinned = false
        activeKey = null
        resetView()
      })
      .on("focus.nativeChart", () => {
        if (activeKey === null && keys.length && !drag && !zoomArmed) inspect(keys[0], true)
      })
      .on("keydown.nativeChart", (event: KeyboardEvent) => {
        if (event.ctrlKey || event.metaKey || event.altKey) return
        if (event.key === "Escape") {
          exitInteraction(event)
          return
        }
        if (event.key === "0") {
          event.preventDefault()
          resetView()
          return
        }
        if (drag) {
          event.preventDefault()
          return
        }
        if (["+", "=", "-", "_"].includes(event.key)) {
          event.preventDefault()
          const factor = event.key === "+" || event.key === "=" ? 0.5 : 2
          view = {
            x: scaleChartRange(view.x, full.x, factor),
            y: categories ? full.y : scaleChartRange(view.y, full.y, factor),
          }
          render()
          announcement.textContent = factor < 1 ? "Chart view zoomed in." : "Chart view zoomed out."
          return
        }
        if (event.shiftKey && event.key.startsWith("Arrow")) {
          event.preventDefault()
          const dx = event.key === "ArrowRight" ? 0.1 : event.key === "ArrowLeft" ? -0.1 : 0
          const dy = event.key === "ArrowUp" ? 0.1 : event.key === "ArrowDown" ? -0.1 : 0
          view = {
            x: panChartRange(view.x, full.x, dx),
            y: categories ? full.y : panChartRange(view.y, full.y, dy),
          }
          render()
          announcement.textContent = "Chart view panned."
          return
        }
        if (!keys.length) return
        let index = activeKey === null ? -1 : keys.indexOf(activeKey)
        switch (event.key) {
          case "Enter":
          case " ":
            event.preventDefault()
            if (event.repeat) return
            pinned = !pinned
            inspect(activeKey ?? keys[0], true)
            return
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
      endCapture(true)
      zoomArmed = false
      width = next
      render()
    }
  })
  resize.observe(plot)
  cleanups.push(() => {
    resize.disconnect()
    endCapture(true)
    svg.on(".nativeChart", null)
    select(content).on(".nativeChart", null)
    for (const button of [pinButton, zoomButton, resetButton])
      select(button).on(".nativeChart", null)
    controls.forEach(({ value }) =>
      value.getAnimations().forEach((animation) => animation.cancel()),
    )
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
    const fallback = content.querySelector<HTMLAnchorElement>(".native-chart-download")!
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
        content.replaceChildren(message, fallback)
        console.warn("Unable to load native chart", figure.dataset.chartSrc, error)
      }
    })()
  }
})
