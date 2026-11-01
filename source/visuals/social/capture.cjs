// NODE_PATH=/path/to/temporary-playwright/node_modules node visuals/social/capture.cjs
// Capture the actual article widgets; no browser dependency is added to the blog.
const { chromium } = require("playwright")
const fs = require("node:fs/promises")
const path = require("node:path")
const { pathToFileURL } = require("node:url")
const http = require("node:http")

const content = path.resolve(__dirname, "../../content")
const output = path.join(content, "media/social")
const site = path.resolve(__dirname, "../../public")
const types = {
  ".html": "text/html",
  ".css": "text/css",
  ".js": "text/javascript",
  ".svg": "image/svg+xml",
}

// Serve the built site (`npm run build`) so pages can run their scripts; file:// cannot.
async function serveSite() {
  const server = http.createServer(async (request, response) => {
    const route = decodeURIComponent(new URL(request.url, "http://localhost").pathname)
    const file = path.join(site, route)
    for (const candidate of [file, `${file}.html`, path.join(file, "index.html")]) {
      if (!candidate.startsWith(site)) break
      const body = await fs.readFile(candidate).catch(() => undefined)
      if (!body) continue
      response.writeHead(200, {
        "content-type": types[path.extname(candidate)] ?? "application/octet-stream",
      })
      return response.end(body)
    }
    response.writeHead(404).end()
  })
  await new Promise((resolve) => server.listen(0, "127.0.0.1", resolve))
  return server
}

async function main() {
  await fs.mkdir(output, { recursive: true })
  const browser = await chromium.launch({ headless: true })
  try {
    const page = await browser.newPage({
      viewport: { width: 1080, height: 286 },
      deviceScaleFactor: 2,
      colorScheme: "light",
      reducedMotion: "reduce",
    })
    await page.goto(pathToFileURL(path.join(content, "media/kda/kda-future-animation.html")).href)
    await page.locator("canvas").evaluate(() => {
      running = false
      draw(0)
    })
    await page.locator("canvas").screenshot({ path: path.join(output, "kda-reference.png") })

    await page.clock.install()
    await page.goto(pathToFileURL(path.join(content, "widgets/ptq-conveyor-board.html")).href)
    await page.evaluate(() => {
      window.requestAnimationFrame = () => 0
      render(2400)
    })
    await page.locator("canvas").screenshot({ path: path.join(output, "ptq-dispatch.png") })

    await page.goto(pathToFileURL(path.join(content, "widgets/ulp-binades.html")).href)
    await page.addStyleTag({
      content:
        "body {margin:0} main {padding:0} #overviewViz {width:1080px;height:286px;max-width:none} .grid {display:block} .panel {padding:0;border:0}",
    })
    await page.locator("#overviewViz").screenshot({ path: path.join(output, "ulp-binades.png") })

    await page.goto(pathToFileURL(path.join(content, "widgets/clc-work-distribution.html")).href)
    for (let i = 0; i < 8; i++) await page.locator("#step").click()
    await page.addStyleTag({
      content:
        "body{margin:0} main{padding:8px 20px;max-width:none} .hero,.queue-panel,.footer-note{display:none} .worker-panel{margin:0;padding:4px 0;border:0} .workers{display:grid;grid-template-columns:repeat(2,1fr);gap:6px 24px} .worker{min-height:0;padding:3px;grid-template-columns:42px minmax(0,1fr) 55px} .lane{min-height:24px;padding:3px;gap:3px} .tile{width:20px;height:20px;font-size:10px} h2{margin:0 0 12px}",
    })
    await page.locator(".worker-panel").screenshot({ path: path.join(output, "clc-workers.png") })
    // Fast math: the explorer's sin heatmaps (CUDA default vs --use_fast_math), rows 2^-23..2^23.
    await page.setViewportSize({ width: 1080, height: 900 })
    await page.goto(
      pathToFileURL(path.join(content, "media/fast-math/fast-math-explorer.html")).href,
    )
    await page.waitForFunction(() => window.__fm)
    await page.evaluate(() => {
      window.__fm.setOp("sin")
      window.__fm.setView(104, 150)
      window.__fm.hideCursor()
    })
    await page.addStyleTag({
      content:
        "body{margin:0} main{padding:0;max-width:none} main>*:not(#fig){display:none!important} #fig{border:0;padding:0} #fig>*:not(.pair){display:none!important} .pair{gap:0} .side>*:not(canvas){display:none!important} canvas.heat{width:540px!important}",
    })
    const pair = await page.locator("#fig .pair").boundingBox()
    await page.screenshot({
      path: path.join(output, "fast-math-sin.png"),
      clip: { x: 0, y: pair.y + (pair.height - 286) / 2, width: 1080, height: 286 },
    })

    // TorchInductor walkthrough: the built page (default dark theme) with commentary beside the
    // pinned log; the `cannot fuse op0 with op1_op2` quote is pinned to highlight its lines.
    const server = await serveSite()
    try {
      const width = 1440
      const inductor = await browser.newPage({
        viewport: { width, height: 900 },
        deviceScaleFactor: 2160 / width,
        colorScheme: "dark",
        reducedMotion: "reduce",
      })
      await inductor.addInitScript(() => localStorage.setItem("nuggets-theme", "dark"))
      await inductor.goto(
        `http://127.0.0.1:${server.address().port}/TorchInductor-Walkthrough.html`,
        { waitUntil: "networkidle" },
      )
      const quote = inductor
        .locator("code.nb-quote", { hasText: "cannot fuse op0 with op1_op2" })
        .first()
      await quote.scrollIntoViewIfNeeded()
      await quote.click()
      await inductor.mouse.move(0, 0)
      await inductor.waitForTimeout(800)
      const box = await quote.boundingBox()
      const height = (width * 286) / 1080
      await inductor.screenshot({
        path: path.join(output, "inductor-walkthrough.png"),
        clip: { x: 0, y: box.y + box.height + 24 - height, width, height },
      })
    } finally {
      server.close()
    }
    console.log(
      "Captured KDA, ptq, ULP, CLC, fast-math, and TorchInductor artwork from the original pages.",
    )
  } finally {
    await browser.close()
  }
}
main().catch((error) => {
  console.error(error)
  process.exitCode = 1
})
