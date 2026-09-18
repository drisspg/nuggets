// Run with Playwright in NODE_PATH; see content/Native Charts.md. No repo browser dependency.
const assert = require("node:assert/strict")
const fs = require("node:fs")
const os = require("node:os")
const path = require("node:path")
const http = require("node:http")
const { execFileSync } = require("node:child_process")
const { chromium } = require("playwright")
const serve = require("serve-handler")

const root = path.resolve(__dirname, "..")
const work = fs.mkdtempSync(path.join(os.tmpdir(), "nuggets-native-charts-"))
const content = path.join(work, "content")
const output = path.join(work, "site")
const names = ["scaled-loss", "scaled-gap", "paired-checkpoints", "paired-seeds"]
const specs = names.map((name) =>
  JSON.parse(fs.readFileSync(path.join(root, "content/media/kda", `${name}.json`), "utf8")),
)
fs.mkdirSync(path.join(content, "media/kda"), { recursive: true })
fs.mkdirSync(path.join(content, "nested"))
for (const name of names)
  fs.copyFileSync(
    path.join(root, "content/media/kda", `${name}.json`),
    path.join(content, "media/kda", `${name}.json`),
  )
const embed = (src, title) =>
  `\n\`\`\`chart\n${JSON.stringify({ src, title, height: 280 })}\n\`\`\`\n`
fs.writeFileSync(
  path.join(content, "index.md"),
  `---\ntitle: Native chart checks\ndate: 2026-09-17\n---\n[Open nested charts](nested/charts)\n${names.map((name) => embed(`media/kda/${name}.json`, name)).join("\n")}`,
)
fs.writeFileSync(
  path.join(content, "nested/charts.md"),
  `---\ntitle: Nested charts\ndate: 2026-09-17\n---\n[Return to chart checks](../index)\n${embed("media/kda/scaled-gap.json", "Nested gap chart")}`,
)
const gapSpec = {
  version: 1,
  x: { label: "X" },
  y: { label: "Y" },
  intervalLabel: "Supplied interval",
  series: [
    {
      name: "<img src=x onerror=alert(1)>",
      mode: "line",
      points: [
        { x: 0, y: 1, yLow: 0.9, yHigh: 1.1 },
        { x: 1, y: null },
        { x: 2, y: 1, details: { Note: "<script>alert(1)</script>" } },
      ],
    },
  ],
}
fs.writeFileSync(path.join(content, "media/gap.json"), JSON.stringify(gapSpec))
fs.writeFileSync(
  path.join(content, "edge.md"),
  `---\ntitle: Edge cases\ndate: 2026-09-17\n---\n${embed("media/gap.json", "Gaps <script>alert(1)</script>")}\n\`\`\`plotly\n{"src":"media/legacy.html","title":"Legacy Plotly embed","height":180}\n\`\`\``,
)
fs.writeFileSync(
  path.join(content, "media/legacy.html"),
  "<!doctype html><title>Legacy</title><p>Legacy embed retained.</p>",
)
fs.mkdirSync(path.join(content, "widgets"))
fs.writeFileSync(
  path.join(content, "widgets/scrollable.html"),
  '<!doctype html><style>html,body{margin:0}main{height:2400px;background:linear-gradient(#ddd,#888)}</style><main>Wheel-driven viewer</main><script>window.widgetWheelCount=0;document.addEventListener("wheel",event=>{event.preventDefault();window.widgetWheelCount++;window.scrollBy({top:event.deltaY,behavior:"instant"})},{passive:false})</script>',
)
fs.writeFileSync(
  path.join(content, "widget-check.md"),
  '---\ntitle: Widget interaction checks\ndate: 2026-09-17\n---\n<div style="height:100vh"></div>\n\n![[widgets/scrollable.html|Scroll test widget]]\n\n<div style="height:100vh"></div>\n',
)
execFileSync(
  process.execPath,
  ["quartz/bootstrap-cli.mjs", "build", "--directory", content, "--output", output],
  { cwd: root, stdio: "inherit", timeout: 60000 },
)

async function main() {
  const server = http.createServer((req, res) => {
    // Exercise a deployed subpath as well as content-relative paths in nested notes.
    req.url = req.url.replace(/^\/blog(?=\/|$)/, "") || "/"
    void serve(req, res, { public: output, cleanUrls: true })
  })
  await new Promise((resolve) => server.listen(0, "127.0.0.1", resolve))
  const base = `http://127.0.0.1:${server.address().port}/blog`
  let browser
  try {
    browser = await chromium.launch()
    for (const width of [320, 390, 960, 1440]) {
      const page = await browser.newPage({ viewport: { width, height: 1000 } })
      const errors = []
      page.on("pageerror", (error) => errors.push(error.message))
      await page.goto(`${base}/`)
      await page.waitForFunction(
        () =>
          document.querySelectorAll(".native-chart-plot svg").length === 4 &&
          document.querySelectorAll(".native-chart-point").length === 62,
      )
      await page.evaluate(() => document.fonts.ready)
      // Focus/scroll setup must settle before measuring hover layout, not animate between samples.
      await page.addStyleTag({ content: "html { scroll-behavior: auto !important; }" })
      const charts = page.locator(".native-chart")
      assert.equal(await charts.count(), 4)
      assert.equal(await page.locator(".plotly-chart__frame").count(), 0)
      for (let i = 0; i < 4; i++) {
        const chart = charts.nth(i)
        assert.equal(await chart.locator(".native-chart-point").count(), [32, 16, 10, 4][i])
        assert.equal(await chart.locator(".native-chart-line").count(), [4, 2, 1, 0][i])
        assert.equal(await chart.locator(".native-chart-error-y").count(), [0, 16, 10, 0][i])
        assert.equal(await chart.locator(".native-chart-error-x").count(), [0, 0, 0, 4][i])
        assert.equal(await chart.locator("tbody tr").count(), [32, 16, 10, 4][i])
        const expected = specs[i].series.flatMap((series) =>
          series.points.map((point) => [String(point.x), String(point.y)]),
        )
        assert.deepEqual(
          await chart
            .locator(".native-chart-point")
            .evaluateAll((nodes) => nodes.map((node) => [node.dataset.x, node.dataset.y])),
          expected,
        )
        await chart.locator("svg").scrollIntoViewIfNeeded()
        await page.mouse.move(0, 0)
        const idlePlotBox = await chart.locator("svg").boundingBox()
        const hoverPoint = await chart.locator(".native-chart-point").first().boundingBox()
        await page.mouse.move(hoverPoint.x + hoverPoint.width / 2, hoverPoint.y + hoverPoint.height / 2)
        const livePlotBox = await chart.locator("svg").boundingBox()
        assert(Math.abs(idlePlotBox.y - livePlotBox.y) < 1, `chart ${i} legend must not rewrap on hover`)
        await chart.locator("svg").focus()
        await page.keyboard.press("End")
        assert.equal(
          await chart.locator(".native-chart-inspection circle").count(),
          [4, 2, 2, 1][i],
        )
        assert(
          (await chart.locator(".native-chart-legend-value").allTextContents()).every(
            (text) => text !== "—",
          ),
        )
        await page.keyboard.press("Enter")
        assert.equal(await chart.getAttribute("data-pinned"), "true")
        await page.keyboard.press("Home")
        assert.equal(await chart.getAttribute("data-pinned"), "true")
        await page.keyboard.press("Space")
        assert.equal(await chart.getAttribute("data-pinned"), "false")
        const escapeScroll = await page.evaluate(() => scrollY)
        await page.keyboard.press("Escape")
        assert.equal(await chart.locator(".native-chart-inspection circle").count(), 0)
        assert(
          Math.abs((await page.evaluate(() => scrollY)) - escapeScroll) < 1,
          "Escape must not jump to the closed search button",
        )
        assert(
          await chart.locator("svg").evaluate((node) => document.activeElement !== node),
          "Escape releases chart keyboard focus",
        )
        await page.keyboard.press("Escape")
        assert(
          Math.abs((await page.evaluate(() => scrollY)) - escapeScroll) < 1,
          "Escape outside an interaction must not move the page",
        )
      }
      const searchOpener = charts.first().locator("svg")
      await searchOpener.focus()
      const beforeSearch = await page.evaluate(() => scrollY)
      await page.keyboard.press("Control+k")
      await page.locator("#search-container.active").waitFor()
      await page.keyboard.press("Escape")
      assert.equal(await page.locator("#search-container.active").count(), 0)
      assert(
        await searchOpener.evaluate((node) => document.activeElement === node),
        "search returns focus to its actual opener",
      )
      assert(Math.abs((await page.evaluate(() => scrollY)) - beforeSearch) < 1)
      await page.keyboard.press("Escape")
      await page.keyboard.press("ArrowDown")
      await page.waitForFunction((before) => scrollY > before, beforeSearch)
      const forest = charts.nth(3)
      await forest.locator(".native-chart-legend-item").focus()
      assert(
        await forest
          .locator(".native-chart-legend-item")
          .evaluate((node) => !node.disabled && document.activeElement === node),
        "single-series legend remains keyboard-accessible",
      )
      await forest.locator("svg").focus()
      await page.keyboard.press("End")
      const finalPoint = specs[3].series[0].points.at(-1)
      const readout = await forest.locator(".native-chart-legend-item").getAttribute("title")
      assert((await forest.locator(".native-chart-axis").first().locator(".tick").count()) >= 2)
      for (const value of [
        finalPoint.x,
        finalPoint.xLow,
        finalPoint.xHigh,
        finalPoint.details["Paired SE"],
      ])
        assert(readout.includes(String(value)))
      const loss = charts.first()
      await loss.locator("svg").scrollIntoViewIfNeeded()
      const beforeHover = await loss.locator("svg").boundingBox()
      // Curves overlap: inspect a coordinate, not whichever marker happens to be on top.
      const firstPoint = await loss
        .locator('.native-chart-point[data-x="1000"]')
        .first()
        .boundingBox()
      await page.mouse.move(
        firstPoint.x + firstPoint.width / 2,
        firstPoint.y + firstPoint.height / 2,
      )
      assert(
        (await loss.locator(".native-chart-legend-item").first().getAttribute("title")).includes(
          String(specs[0].series[0].points[0].y),
        ),
      )
      const afterHover = await loss.locator("svg").boundingBox()
      assert(
        Math.abs(beforeHover.y - afterHover.y) < 1,
        `legend updates moved the plot: before=${JSON.stringify(beforeHover)}, after=${JSON.stringify(afterHover)}`,
      )
      await page.mouse.click(
        firstPoint.x + firstPoint.width / 2,
        firstPoint.y + firstPoint.height / 2,
      )
      assert.equal(await loss.getAttribute("data-pinned"), "true")
      const pinnedValues = await loss.locator(".native-chart-legend-value").allTextContents()
      const pinnedCrosshair = await loss.locator(".native-chart-crosshair").getAttribute("x1")
      const lastPoint = await loss
        .locator('.native-chart-point[data-x="7600"]')
        .first()
        .boundingBox()
      await page.mouse.move(lastPoint.x + lastPoint.width / 2, lastPoint.y + lastPoint.height / 2)
      assert.deepEqual(
        await loss.locator(".native-chart-legend-value").allTextContents(),
        pinnedValues,
      )
      assert.equal(
        await loss.locator(".native-chart-crosshair").getAttribute("x1"),
        pinnedCrosshair,
      )
      await page.mouse.click(lastPoint.x + lastPoint.width / 2, lastPoint.y + lastPoint.height / 2)
      assert.equal(await loss.getAttribute("data-pinned"), "false")
      assert.notDeepEqual(
        await loss.locator(".native-chart-legend-value").allTextContents(),
        pinnedValues,
      )
      await page.keyboard.press("Enter")
      assert.equal(await loss.getAttribute("data-pinned"), "true")
      await page.keyboard.press("Escape")
      assert.equal(await loss.getAttribute("data-pinned"), "false")
      const beforeAxes = await loss.locator(".native-chart-axis").allTextContents()
      const buttons = loss.locator(".native-chart-legend-item")
      await buttons.nth(0).click()
      assert.equal(await loss.locator(".native-chart-line").count(), 3)
      assert.deepEqual(await loss.locator(".native-chart-axis").allTextContents(), beforeAxes)
      assert.equal(await loss.locator("tbody tr").count(), 32)
      await buttons.nth(1).click()
      await buttons.nth(2).click()
      assert(await buttons.nth(3).isDisabled())
      await buttons.nth(0).click()
      await buttons.nth(1).click()
      await buttons.nth(2).click()
      assert.equal(await loss.locator(".native-chart-line").count(), 4)
      const sweep = charts.nth(2)
      await sweep.locator("svg").focus()
      await page.keyboard.press("Home")
      await page.keyboard.press("Enter")
      assert.equal(await sweep.getAttribute("data-pinned"), "true")
      await sweep.locator(".native-chart-legend-item").first().click()
      assert.equal(
        await sweep.getAttribute("data-pinned"),
        "false",
        "hiding the only observation at a pinned coordinate releases it",
      )
      await sweep.locator(".native-chart-legend-item").first().click()
      await loss.locator("svg").focus()
      await page.keyboard.press("End")
      await page.keyboard.press("Enter")
      const themedValues = await loss.locator(".native-chart-legend-value").allTextContents()
      for (const theme of ["light", "dark"]) {
        if ((await page.locator("html").getAttribute("saved-theme")) !== theme)
          await page.locator(".darkmode").click()
        await loss.scrollIntoViewIfNeeded()
        const color = await loss
          .locator(".native-chart-series")
          .first()
          .evaluate((node) => getComputedStyle(node).color)
        assert.equal(color, theme === "light" ? "rgb(23, 109, 156)" : "rgb(104, 181, 230)")
        assert.equal(await loss.getAttribute("data-pinned"), "true")
        assert.deepEqual(
          await loss.locator(".native-chart-legend-value").allTextContents(),
          themedValues,
        )
        await page.waitForTimeout(180)
        assert(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth))
        await loss.screenshot({ path: path.join(work, `loss-${width}-${theme}.png`) })
        await forest.screenshot({ path: path.join(work, `forest-${width}-${theme}.png`) })
      }
      await loss.locator("svg").scrollIntoViewIfNeeded()
      const box = await loss.locator("svg").boundingBox()
      await page.mouse.move(box.x + box.width * 0.6, box.y + 100)
      const scroll = await page.evaluate(() => scrollY)
      await page.mouse.wheel(0, 300)
      await page.waitForFunction((before) => scrollY > before, scroll)
      // Repeated SPA returns should not duplicate SVGs or toggle listeners.
      for (let visit = 0; visit < 2; visit++) {
        await page.getByRole("link", { name: "Open nested charts" }).click()
        await page.waitForURL("**/nested/charts")
        await page.waitForFunction(
          () => document.querySelectorAll(".native-chart-point").length === 16,
        )
        assert.equal(
          await page.locator(".native-chart-download").evaluate((node) => node.href),
          `${base}/media/kda/scaled-gap.json`,
        )
        await page.getByRole("link", { name: "Return to chart checks" }).click()
        await page.waitForURL("**/blog/")
        await page.waitForFunction(
          () => document.querySelectorAll(".native-chart-point").length === 62,
        )
      }
      await page.locator(".native-chart-legend-item").first().click()
      assert.equal(
        await page.locator(".native-chart").first().locator(".native-chart-line").count(),
        3,
      )
      const resizedChart = page.locator(".native-chart").first()
      await resizedChart.locator("svg").focus()
      await page.keyboard.press("Home")
      await page.keyboard.press("Enter")
      const beforeResize = await resizedChart
        .locator(".native-chart-legend-value")
        .allTextContents()
      await page.setViewportSize({ width: width === 1440 ? 390 : 1440, height: 1000 })
      await page.waitForTimeout(100)
      assert.equal(await resizedChart.getAttribute("data-pinned"), "true")
      assert.deepEqual(
        await resizedChart.locator(".native-chart-legend-value").allTextContents(),
        beforeResize,
      )
      assert(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth))
      assert.deepEqual(errors, [])
      console.log(
        `${width}px: source parity, intervals, keyboard, legends, themes, wheel, subpath/SPA and resize passed`,
      )
      await page.close()
    }
    const widgetPage = await browser.newPage({ viewport: { width: 960, height: 1000 } })
    await widgetPage.goto(`${base}/widget-check`)
    await widgetPage.addStyleTag({ content: "html { scroll-behavior: auto !important; }" })
    const shell = widgetPage.locator(".widget-shell")
    const widget = shell.locator("iframe")
    await shell.scrollIntoViewIfNeeded()
    const widgetFrame = await (await widget.elementHandle()).contentFrame()
    await widgetFrame.locator("main").waitFor()
    assert.equal(await widget.getAttribute("tabindex"), "-1")
    let widgetBox = await widget.boundingBox()
    await widgetPage.mouse.move(
      widgetBox.x + widgetBox.width / 2,
      widgetBox.y + widgetBox.height / 2,
    )
    const passiveScroll = await widgetPage.evaluate(() => scrollY)
    await widgetPage.mouse.wheel(0, 150)
    await widgetPage.waitForFunction((before) => scrollY > before, passiveScroll)
    assert.equal(await widgetFrame.evaluate(() => scrollY), 0)
    assert.equal(await widgetFrame.evaluate(() => window.widgetWheelCount), 0)
    await shell.locator(".widget-activate").click()
    assert(await shell.evaluate((node) => node.classList.contains("widget-active")))
    assert.equal(await widget.getAttribute("tabindex"), "0")
    widgetBox = await widget.boundingBox()
    await widgetPage.mouse.move(
      widgetBox.x + widgetBox.width / 2,
      widgetBox.y + widgetBox.height / 2,
    )
    const activeScroll = await widgetPage.evaluate(() => scrollY)
    await widgetPage.mouse.wheel(0, 150)
    await widgetFrame.waitForFunction(() => window.widgetWheelCount > 0 && scrollY > 0)
    assert(Math.abs((await widgetPage.evaluate(() => scrollY)) - activeScroll) < 1)
    await widgetPage.keyboard.press("Escape")
    assert(!(await shell.evaluate((node) => node.classList.contains("widget-active"))))
    assert.equal(await widget.getAttribute("tabindex"), "-1")
    assert(Math.abs((await widgetPage.evaluate(() => scrollY)) - activeScroll) < 1)
    assert(
      await shell.locator(".widget-activate").evaluate((node) => document.activeElement === node),
    )
    await widgetPage.mouse.wheel(0, 150)
    await widgetPage.waitForFunction((before) => scrollY > before, activeScroll)
    await widgetPage.close()
    console.log(
      "Inactive widgets pass scroll to the page; activation opts in; iframe Escape releases in place",
    )

    const touch = await browser.newPage({
      viewport: { width: 390, height: 900 },
      isMobile: true,
      hasTouch: true,
    })
    await touch.goto(`${base}/`)
    await touch.waitForSelector(".native-chart-point")
    const touchPlot = touch.locator(".native-chart-plot svg").first()
    await touchPlot.scrollIntoViewIfNeeded()
    const touchBox = await touchPlot.boundingBox()
    const startScroll = await touch.evaluate(() => scrollY)
    const cdp = await touch.context().newCDPSession(touch)
    const touchPoint = { x: touchBox.x + touchBox.width / 2, y: touchBox.y + 210 }
    await cdp.send("Input.dispatchTouchEvent", { type: "touchStart", touchPoints: [touchPoint] })
    for (let step = 1; step <= 5; step++) {
      await cdp.send("Input.dispatchTouchEvent", {
        type: "touchMove",
        touchPoints: [{ ...touchPoint, y: touchPoint.y - step * 30 }],
      })
      await touch.waitForTimeout(20)
    }
    await cdp.send("Input.dispatchTouchEvent", { type: "touchEnd", touchPoints: [] })
    await touch.waitForFunction((before) => scrollY > before, startScroll)
    assert.equal(
      await touch.locator(".native-chart-inspection circle").count(),
      0,
      "vertical touch scrolling must not select data",
    )
    await cdp.detach()
    await touch.goto(`${base}/`)
    await touch.waitForSelector(".native-chart-point")
    await touchPlot.tap({ position: { x: 180, y: 100 } })
    assert((await touch.locator(".native-chart-inspection circle").count()) > 0)
    assert.equal(await touch.locator(".native-chart").first().getAttribute("data-pinned"), "true")
    await touchPlot.tap({ position: { x: 220, y: 120 } })
    assert.equal(await touch.locator(".native-chart").first().getAttribute("data-pinned"), "false")
    await touch.close()

    const edge = await browser.newPage()
    const dialogs = []
    edge.on("dialog", (dialog) => {
      dialogs.push(dialog.message())
      void dialog.dismiss()
    })
    await edge.goto(`${base}/edge`)
    await edge.waitForSelector(".native-chart-point")
    assert.equal(await edge.locator(".native-chart-point").count(), 2)
    assert.equal((await edge.locator(".native-chart-line").getAttribute("d")).match(/M/g).length, 2)
    assert.equal(await edge.locator(".native-chart img").count(), 0)
    assert.equal(await edge.locator(".plotly-chart__frame").count(), 1)
    assert((await edge.locator(".native-chart-data").textContent()).includes("Missing"))
    await edge.locator(".native-chart-plot svg").focus()
    await edge.keyboard.press("End")
    assert(
      (await edge.locator(".native-chart-legend-item").getAttribute("title")).includes(
        "<script>alert(1)</script>",
      ),
    )
    await edge.keyboard.press("Home")
    await edge.keyboard.press("ArrowRight")
    assert.equal(await edge.locator(".native-chart-legend-value").innerText(), "Missing")
    assert.equal(await edge.locator(".native-chart-inspection circle").count(), 0)
    assert.equal(
      await edge.locator("tbody tr").nth(2).locator("td").nth(3).textContent(),
      "—",
      "no interval is distinct from a missing observation",
    )
    assert.deepEqual(dialogs, [])
    await edge.close()

    for (const badResponse of [
      { status: 503, body: "Unavailable" },
      { status: 200, contentType: "application/json", body: '{"version":999}' },
    ]) {
      const page = await browser.newPage()
      await page.route("**/scaled-loss.json", (route) => route.fulfill(badResponse))
      await page.goto(`${base}/`)
      const chart = page.locator(".native-chart").first()
      await chart.locator('[role="alert"]').waitFor()
      assert(await chart.locator(".native-chart-download").isVisible())
      assert.equal(await chart.locator(".native-chart-point").count(), 0)
      await page.close()
    }
    const reduced = await browser.newPage({ reducedMotion: "reduce" })
    await reduced.goto(`${base}/`)
    await reduced.waitForSelector(".native-chart-point")
    const reducedChart = reduced.locator(".native-chart").first()
    await reducedChart.locator("svg").focus()
    await reduced.keyboard.press("End")
    await reduced.keyboard.press("Enter")
    assert.equal(await reducedChart.getAttribute("data-pinned"), "true")
    assert(
      await reducedChart
        .locator(".native-chart-legend-value")
        .evaluateAll((nodes) =>
          nodes.every(
            (node) =>
              node.getAnimations().length === 0 &&
              getComputedStyle(node).transitionDuration === "0s",
          ),
        ),
    )
    await reduced.close()

    const noScript = await browser.newPage({ javaScriptEnabled: false })
    await noScript.goto(`${base}/`)
    assert.equal(await noScript.locator(".native-chart-download").count(), 4)
    assert.equal(await noScript.locator(".native-chart-plot").count(), 0)
    assert(
      (await noScript.locator(".native-chart").first().innerText()).includes("requires JavaScript"),
    )
    const sourceResponse = await noScript.request.get(
      await noScript
        .locator(".native-chart-download")
        .first()
        .evaluate((node) => node.href),
    )
    assert.deepEqual(await sourceResponse.json(), specs[0])
    await noScript.close()
    console.log(
      "Touch pan/tap, null gaps, escaped content, legacy Plotly, no-JS and network/schema failure fallback passed",
    )
    console.log(`Browser artifacts: ${work}`)
  } finally {
    await browser?.close()
    server.closeAllConnections()
    await new Promise((resolve) => server.close(resolve))
  }
}
main().catch((error) => {
  console.error(error)
  console.error(`Artifacts retained: ${work}`)
  process.exitCode = 1
})
