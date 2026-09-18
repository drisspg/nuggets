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
  series: [
    {
      name: "<img src=x onerror=alert(1)>",
      mode: "line",
      points: [
        { x: 0, y: 1 },
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
  const browser = await chromium.launch()
  try {
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
        await chart.locator("svg").focus()
        await page.keyboard.press("End")
        assert.equal(
          await chart.locator(".native-chart-inspection circle").count(),
          [4, 2, 2, 1][i],
        )
        assert(await chart.locator(".native-chart-readout").innerText())
        await page.keyboard.press("Escape")
        assert.equal(await chart.locator(".native-chart-inspection circle").count(), 0)
      }
      const forest = charts.nth(3)
      await forest.locator("svg").focus()
      await page.keyboard.press("End")
      const finalPoint = specs[3].series[0].points.at(-1)
      const readout = await forest.locator(".native-chart-readout").innerText()
      for (const value of [
        finalPoint.x,
        finalPoint.xLow,
        finalPoint.xHigh,
        finalPoint.details["Paired SE"],
      ])
        assert(readout.includes(String(value)))
      const loss = charts.first()
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
      for (const theme of ["light", "dark"]) {
        if ((await page.locator("html").getAttribute("saved-theme")) !== theme)
          await page.locator(".darkmode").click()
        await loss.scrollIntoViewIfNeeded()
        const color = await loss
          .locator(".native-chart-series")
          .first()
          .evaluate((node) => getComputedStyle(node).color)
        assert.equal(color, theme === "light" ? "rgb(23, 109, 156)" : "rgb(104, 181, 230)")
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
      await page.setViewportSize({ width: width === 1440 ? 390 : 1440, height: 1000 })
      await page.waitForTimeout(100)
      assert(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth))
      assert.deepEqual(errors, [])
      console.log(
        `${width}px: source parity, intervals, keyboard, legends, themes, wheel, subpath/SPA and resize passed`,
      )
      await page.close()
    }
    const touch = await browser.newPage({
      viewport: { width: 390, height: 900 },
      isMobile: true,
      hasTouch: true,
    })
    await touch.goto(`${base}/`)
    await touch.waitForSelector(".native-chart-point")
    const touchPlot = touch.locator(".native-chart-plot svg").first()
    await touchPlot.tap({ position: { x: 180, y: 100 } })
    assert((await touch.locator(".native-chart-inspection circle").count()) > 0)
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
      (await edge.locator(".native-chart-readout").textContent()).includes(
        "<script>alert(1)</script>",
      ),
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
    console.log(
      "Touch, null gaps, escaped content, legacy Plotly, network/schema failure fallback passed",
    )
    console.log(`Browser artifacts: ${work}`)
  } finally {
    await browser.close()
    server.closeAllConnections()
    await new Promise((resolve) => server.close(resolve))
  }
}
main().catch((error) => {
  console.error(error)
  console.error(`Artifacts retained: ${work}`)
  process.exitCode = 1
})
