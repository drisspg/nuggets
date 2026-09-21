// NODE_PATH=/path/to/temporary-playwright/node_modules node visuals/social/capture.cjs
// Capture the actual article widgets; no browser dependency is added to the blog.
const { chromium } = require("playwright")
const fs = require("node:fs/promises")
const path = require("node:path")
const { pathToFileURL } = require("node:url")

const content = path.resolve(__dirname, "../../content")
const output = path.join(content, "media/social")

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
    console.log("Captured KDA, ptq, ULP, and CLC artwork from the original widgets.")
  } finally {
    await browser.close()
  }
}
main().catch((error) => {
  console.error(error)
  process.exitCode = 1
})
