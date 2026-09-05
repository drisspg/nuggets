/** Lazy trace viewers and same-origin Plotly theming, scoped to Quartz navigation. */
interface PlotlyGraph extends HTMLElement {
  layout?: Record<string, unknown>
}
interface PlotlyWindow extends Window {
  Plotly?: {
    relayout: (plot: PlotlyGraph, update: Record<string, unknown>) => Promise<unknown>
    Plots: { resize: (plot: PlotlyGraph) => void }
  }
}

document.addEventListener("nav", () => {
  const controller = new AbortController()
  const { signal } = controller
  const timers = new Set<ReturnType<typeof setTimeout>>()
  window.addCleanup(() => {
    controller.abort()
    timers.forEach(clearTimeout)
    document.body.classList.remove("trace-embed-fullscreen-open")
  })

  for (const embed of document.querySelectorAll<HTMLElement>("[data-trace-embed]")) {
    const preview = embed.querySelector<HTMLButtonElement>(".trace-preview")!
    const viewer = embed.querySelector<HTMLElement>(".trace-embed__viewer")!
    const frame = viewer.querySelector<HTMLIFrameElement>("iframe")!
    const close = viewer.querySelector<HTMLButtonElement>("[data-trace-close]")!
    const fullscreen = viewer.querySelector<HTMLButtonElement>("[data-trace-fullscreen]")!
    /** Keep native and CSS fullscreen controls in sync. */
    const syncFullscreen = () => {
      const active =
        document.fullscreenElement === viewer ||
        viewer.classList.contains("trace-embed__viewer--fullscreen")
      fullscreen.textContent = active ? "Exit fullscreen" : "Fullscreen"
      fullscreen.setAttribute("aria-pressed", String(active))
    }
    /** Leave either fullscreen mode before hiding the viewer. */
    const leaveFullscreen = async () => {
      if (document.fullscreenElement === viewer) await document.exitFullscreen()
      viewer.classList.remove("trace-embed__viewer--fullscreen")
      document.body.classList.remove("trace-embed-fullscreen-open")
      syncFullscreen()
    }
    preview.addEventListener(
      "click",
      () => {
        preview.hidden = true
        preview.setAttribute("aria-expanded", "true")
        viewer.hidden = false
        if (!frame.hasAttribute("src")) frame.src = frame.dataset.src!
        close.focus()
      },
      { signal },
    )
    close.addEventListener(
      "click",
      async () => {
        await leaveFullscreen()
        viewer.hidden = true
        preview.hidden = false
        preview.setAttribute("aria-expanded", "false")
        preview.focus()
      },
      { signal },
    )
    fullscreen.addEventListener(
      "click",
      async () => {
        if (
          document.fullscreenElement === viewer ||
          viewer.classList.contains("trace-embed__viewer--fullscreen")
        ) {
          await leaveFullscreen()
        } else {
          try {
            await viewer.requestFullscreen()
          } catch {
            if (signal.aborted) return
            viewer.classList.add("trace-embed__viewer--fullscreen")
            document.body.classList.add("trace-embed-fullscreen-open")
          }
          syncFullscreen()
        }
      },
      { signal },
    )
    document.addEventListener("fullscreenchange", syncFullscreen, { signal })
    document.addEventListener(
      "keydown",
      (event) => {
        if (
          event.key === "Escape" &&
          viewer.classList.contains("trace-embed__viewer--fullscreen")
        ) {
          void leaveFullscreen()
          fullscreen.focus()
        }
      },
      { signal },
    )
  }

  for (const frame of document.querySelectorAll<HTMLIFrameElement>(".plotly-chart__frame")) {
    let generation = 0
    let queue = Promise.resolve()
    /** Theme only the chart chrome; preserve data colors, titles, annotations and layout. */
    const styleFrame = async (attempt: number, current: number): Promise<void> => {
      if (signal.aborted || current !== generation || !frame.isConnected) return
      let doc: Document | null
      try {
        doc = frame.contentDocument
      } catch {
        return
      }
      const win = frame.contentWindow as PlotlyWindow | null
      const plots = doc?.querySelectorAll<PlotlyGraph>(".plotly-graph-div")
      if (!doc?.body || !win?.Plotly || !plots?.length || [...plots].some((plot) => !plot.layout)) {
        // Standalone exports may finish Plotly.newPlot after the iframe load event.
        if (attempt < 50) {
          const timer = setTimeout(() => {
            timers.delete(timer)
            enqueue(attempt + 1, current)
          }, 100)
          timers.add(timer)
        }
        return
      }
      const theme = getComputedStyle(document.documentElement)
      const foreground = theme.getPropertyValue("--dark").trim()
      const background = theme.getPropertyValue("--light").trim()
      const grid = theme.getPropertyValue("--lightgray").trim()
      doc.documentElement.style.background = background
      doc.body.style.background = background
      doc.body.style.margin = "0"
      for (const plot of plots) {
        const update: Record<string, unknown> = {
          paper_bgcolor: background,
          plot_bgcolor: background,
          "font.color": foreground,
          "hoverlabel.bgcolor": background,
          "hoverlabel.font.color": foreground,
        }
        for (const key of Object.keys(plot.layout!)) {
          if (/^[xyz]axis\d*$/.test(key)) {
            update[`${key}.gridcolor`] = grid
            update[`${key}.zerolinecolor`] = grid
            update[`${key}.color`] = foreground
          }
        }
        await win.Plotly.relayout(plot, update)
        if (signal.aborted) return
        win.Plotly.Plots.resize(plot)
      }
    }
    /** Serialize relayouts so a theme toggle cannot race an iframe load. */
    const enqueue = (attempt = 0, current = ++generation) => {
      queue = queue
        .then(() => styleFrame(attempt, current))
        .catch((error) => console.warn("Unable to theme Plotly embed", error))
    }
    frame.addEventListener("load", () => enqueue(), { signal })
    document.addEventListener("themechange", () => enqueue(), { signal })
    enqueue()
  }
})
