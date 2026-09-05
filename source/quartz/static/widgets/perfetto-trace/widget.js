const PERFETTO_ORIGIN = "https://ui.perfetto.dev"
const HANDSHAKE_TIMEOUT_MS = 30_000
const params = new URLSearchParams(window.location.search)
const iframe = document.querySelector("#perfetto")
const status = document.querySelector("#status")

/** Display loading failures in the existing status overlay. */
function showError(error) {
  status.textContent = error instanceof Error ? error.message : String(error)
  status.classList.add("trace-viewer__status--error")
}

/** Wait for the embedded UI, releasing handshake resources on every exit. */
function waitForPerfetto(signal) {
  return new Promise((resolve, reject) => {
    const interval = window.setInterval(() => {
      iframe.contentWindow.postMessage("PING", PERFETTO_ORIGIN)
    }, 100)
    const timeout = window.setTimeout(() => {
      cleanup()
      reject(new Error("Timed out waiting for Perfetto"))
    }, HANDSHAKE_TIMEOUT_MS)

    /** Release timers and listeners after success, failure, or cancellation. */
    function cleanup() {
      window.clearInterval(interval)
      window.clearTimeout(timeout)
      window.removeEventListener("message", onMessage)
      signal.removeEventListener("abort", onAbort)
    }

    /** Cancel the handshake when loading fails or the page unloads. */
    function onAbort() {
      cleanup()
      reject(signal.reason)
    }

    /** Accept readiness only from this embedded Perfetto window. */
    function onMessage(event) {
      if (
        event.origin === PERFETTO_ORIGIN &&
        event.source === iframe.contentWindow &&
        event.data === "PONG"
      ) {
        cleanup()
        resolve()
      }
    }

    window.addEventListener("message", onMessage)
    signal.addEventListener("abort", onAbort, { once: true })
    if (signal.aborted) onAbort()
  })
}

/** Fetch and validate the trace independently of Perfetto readiness. */
async function fetchTrace(traceUrl, signal) {
  const response = await fetch(traceUrl, { signal })
  if (!response.ok) {
    throw new Error(`Unable to load trace: ${response.status}`)
  }
  return response.arrayBuffer()
}

/** Load a same-origin trace and transfer it using Perfetto's embed protocol. */
async function loadTrace() {
  const traceParameter = params.get("trace")
  if (!traceParameter) {
    throw new Error("Missing trace query parameter")
  }

  const traceUrl = new URL(traceParameter, window.location.href)
  if (traceUrl.origin !== window.location.origin) {
    throw new Error("Trace URL must use the documentation origin")
  }

  const title = params.get("title") || "Perfetto trace"
  document.title = title
  iframe.title = title
  iframe.src = `${PERFETTO_ORIGIN}/#!/?mode=embedded`

  const controller = new AbortController()
  const onUnload = () => controller.abort()
  window.addEventListener("pagehide", onUnload)
  window.addEventListener("unload", onUnload)
  try {
    const [buffer] = await Promise.all([
      fetchTrace(traceUrl, controller.signal),
      waitForPerfetto(controller.signal),
    ])
    iframe.contentWindow.postMessage(
      {
        perfetto: {
          buffer,
          title,
          fileName: traceUrl.pathname.split("/").at(-1),
        },
      },
      PERFETTO_ORIGIN,
      [buffer],
    )
    status.hidden = true
  } finally {
    controller.abort()
    window.removeEventListener("pagehide", onUnload)
    window.removeEventListener("unload", onUnload)
  }
}

loadTrace().catch(showError)
