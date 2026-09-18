// Ephemeral Playwright setup: see content/Code Annotations.md.
const assert = require("node:assert/strict")
const fs = require("node:fs")
const os = require("node:os")
const path = require("node:path")
const http = require("node:http")
const { execFileSync } = require("node:child_process")
const { chromium } = require("playwright")
const serve = require("serve-handler")

const root = path.resolve(__dirname, "..")
const work = fs.mkdtempSync(path.join(os.tmpdir(), "nuggets-code-annotations-"))
const content = path.join(work, "content")
const output = path.join(work, "site")
fs.mkdirSync(path.join(content, "nested"), { recursive: true })
const guide = fs
  .readFileSync(path.join(root, "content/Code Annotations.md"), "utf8")
  .replace("draft: true\n", "")
fs.writeFileSync(path.join(content, "annotated.md"), guide + "\n[Another page](nested/next)\n")
fs.writeFileSync(
  path.join(content, "index.md"),
  "---\ntitle: Annotation checks\n---\n[Annotations](annotated)\n",
)
fs.writeFileSync(
  path.join(content, "nested/next.md"),
  "---\ntitle: Another page\n---\n[Back to annotations](../annotated)\n",
)
execFileSync(
  process.execPath,
  ["quartz/bootstrap-cli.mjs", "build", "--directory", content, "--output", output],
  { cwd: root, stdio: "inherit", timeout: 60000 },
)

async function main() {
  const server = http.createServer((req, res) => {
    req.url = req.url.replace(/^\/blog(?=\/|$)/, "") || "/"
    void serve(req, res, { public: output, cleanUrls: true })
  })
  await new Promise((resolve) => server.listen(0, "127.0.0.1", resolve))
  const base = `http://127.0.0.1:${server.address().port}/blog`
  let browser
  try {
    browser = await chromium.launch()
    for (const width of [390, 1440]) {
      const context = await browser.newContext({
        viewport: { width, height: 950 },
        permissions: ["clipboard-read", "clipboard-write"],
      })
      const page = await context.newPage()
      const errors = []
      page.on("pageerror", (e) => errors.push(e.message))
      await page.goto(`${base}/annotated`)
      await page.waitForSelector("button.code-annotation-marker")
      await page.evaluate(() => document.fonts.ready)
      await page.addStyleTag({ content: "html { scroll-behavior: auto !important; }" })
      const blocks = page.locator(".code-annotations")
      assert.equal(await blocks.count(), 2)
      assert.equal(await page.locator("button.code-annotation-marker").count(), 5)
      const first = blocks.first()
      const markers = first.locator("button.code-annotation-marker")
      const popup = first.locator(".code-annotation-popover")
      assert.equal(await first.locator("[data-highlighted-line]").count(), 1)
      assert((await first.locator("pre span[style]").count()) > 3, "syntax colors survive")
      const ids = await page
        .locator(".code-annotations [id]")
        .evaluateAll((nodes) => nodes.map((n) => n.id))
      assert.equal(new Set(ids).size, ids.length)

      for (const theme of ["light", "dark"]) {
        await page.evaluate(
          (value) => document.documentElement.setAttribute("saved-theme", value),
          theme,
        )
        await markers.nth(1).click()
        assert(await popup.isVisible())
        assert.match(await popup.innerText(), /Both operands are rescaled/)
        assert((await popup.locator(".katex").count()) > 0)
        assert.equal(await first.locator("[data-code-active]").count(), 2)
        const box = await popup.boundingBox()
        assert(
          box.x >= 0 && box.x + box.width <= width + 1 && box.y >= 0 && box.y + box.height <= 951,
        )
        await page.screenshot({ path: path.join(work, `annotation-${width}-${theme}.png`) })
        await page.keyboard.press("Escape")
        assert.equal(await popup.isVisible(), false)
        assert(await markers.nth(1).evaluate((node) => node === document.activeElement))
        assert.equal(await first.locator("[data-code-active]").count(), 0)
      }
      await markers.first().focus()
      await page.keyboard.press("Space")
      assert(await popup.isVisible())
      await popup.getByRole("button", { name: "Close code annotation" }).click()
      assert.equal(await popup.isVisible(), false)
      assert(await markers.first().evaluate((node) => node === document.activeElement))
      await page.keyboard.press("Enter")
      assert(await popup.isVisible())
      await markers.first().click()
      assert.equal(await popup.isVisible(), false, "same marker toggles its popover closed")
      await markers.first().click()
      await page.getByRole("heading", { name: "Code Annotations", exact: true }).click()
      assert.equal(await popup.isVisible(), false, "outside click dismisses")
      const source = await first.getAttribute("data-code-source")
      await first.locator("pre").first().hover()
      await first.locator("pre .clipboard-button").first().click()
      assert.equal(await page.evaluate(() => navigator.clipboard.readText()), source)
      assert(!source.includes("(1)!"))
      assert.equal(
        await page
          .locator("pre")
          .first()
          .innerText()
          .then((text) => text.includes("# (1)!")),
        true,
        "unannotated literal Markdown example stays intact",
      )

      await markers.first().click()
      await page.evaluate(() =>
        window.scrollTo({ top: document.documentElement.scrollHeight, behavior: "instant" }),
      )
      await page.waitForFunction(
        () => !document.querySelector(".code-annotation-popover:popover-open"),
      )
      assert(
        await markers.first().evaluate((node) => node === document.activeElement),
        "scroll dismissal restores focus rather than dropping it onto the page body",
      )

      // Navigation with a note open must remove the top-layer UI and all listeners.
      for (let repeat = 0; repeat < 2; repeat++) {
        await page.locator("button.code-annotation-marker").first().click()
        await page.getByRole("link", { name: "Another page", exact: true }).click()
        await page.waitForURL(/\/nested\/next$/)
        assert.equal(await page.locator(".code-annotation-popover").count(), 0)
        await page.getByRole("link", { name: "Back to annotations", exact: true }).click()
        await page.waitForURL(/\/annotated$/)
        await page.waitForSelector("button.code-annotation-marker")
        assert.equal(await page.locator("button.code-annotation-marker").count(), 5)
      }
      assert.deepEqual(errors, [])
      console.log(
        `${width}px: themes, rich notes, highlight, keyboard/dismissal, copy, and SPA passed`,
      )
      await context.close()
    }
    const touch = await browser.newContext({
      viewport: { width: 390, height: 844 },
      isMobile: true,
      hasTouch: true,
    })
    const mobile = await touch.newPage()
    await mobile.goto(`${base}/annotated`)
    await mobile.locator("button.code-annotation-marker").first().tap()
    assert(await mobile.locator(".code-annotation-popover:popover-open").isVisible())
    await mobile.getByRole("button", { name: "Close code annotation" }).tap()
    assert.equal(await mobile.locator(".code-annotation-popover:popover-open").count(), 0)
    await touch.close()
    for (const js of [false, true]) {
      const fallback = await browser.newContext({ javaScriptEnabled: js })
      if (js)
        await fallback.addInitScript(() => {
          HTMLElement.prototype.showPopover = undefined
        })
      const page = await fallback.newPage()
      await page.goto(`${base}/annotated`)
      const marker = page.locator("a.code-annotation-marker").first()
      const target = await marker.getAttribute("href")
      await marker.click()
      assert(await page.locator(target).isVisible())
      assert.equal(await page.locator("button.code-annotation-marker").count(), 0)
      await fallback.close()
    }
    console.log("Touch and no-JS/unsupported-popover fallbacks passed")
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
