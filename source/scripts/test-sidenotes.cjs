// Ephemeral Playwright setup: see content/Sidenotes.md.
const assert = require("node:assert/strict")
const fs = require("node:fs")
const os = require("node:os")
const path = require("node:path")
const http = require("node:http")
const { execFileSync } = require("node:child_process")
const { chromium } = require("playwright")
const serve = require("serve-handler")

const root = path.resolve(__dirname, "..")
const work = fs.mkdtempSync(path.join(os.tmpdir(), "nuggets-sidenotes-"))
const content = path.join(work, "content")
const output = path.join(work, "site")
fs.mkdirSync(content)
const text = "A long margin note must not collide with the next note or cover the footer. "
const pair = (id, body) =>
  `<span class="sidenote-pair"><span class="sidenote-ref" tabindex="0" aria-describedby="${id}">More context</span><span class="sidenote" role="note" id="${id}">${body}</span></span>`
fs.writeFileSync(
  path.join(content, "index.md"),
  `---
title: Sidenote layout checks
cssclasses: [sidenotes]
---
First short paragraph. ${pair("long-note", text.repeat(12) + "$\\sqrt{D}\\,\\tau e^{75}$.")}

Second short paragraph. ${pair("next-note", text.repeat(3))}

<details id="extra"><summary>Additional context</summary><p>${pair("expanded-note", text.repeat(5))}</p></details>

A numbered note.<span class="sidenote-ref" aria-hidden="true"></span><span class="sidenote" role="note" id="numbered-note">${text.repeat(2)}</span>

<span class="sidenote-hover"><button class="sidenote-trigger" type="button">Media preview</button><span class="sidenote sidenote--hover-media" role="note">A hover-only media caption.</span></span>

[Another page](next)
`,
)
fs.writeFileSync(path.join(content, "next.md"), "---\ntitle: Another page\n---\n[Back](index)\n")
execFileSync(
  process.execPath,
  ["quartz/bootstrap-cli.mjs", "build", "--directory", content, "--output", output],
  { cwd: root, stdio: "inherit", timeout: 60000 },
)

async function checkLayout(page, width) {
  const boxes = await page
    .locator("article.sidenotes .sidenote:not(.sidenote--hover-media)")
    .evaluateAll((notes) =>
      notes
        .filter((note) => note.checkVisibility())
        .map((note) => {
          const r = note.getBoundingClientRect()
          return {
            id: note.id,
            top: r.top,
            bottom: r.bottom,
            left: r.left,
            right: r.right,
            height: r.height,
          }
        })
        .filter((r) => r.height > 0),
    )
  for (let i = 1; i < boxes.length; i++) {
    assert(
      boxes[i].top >= boxes[i - 1].bottom + 8,
      `${width}px: ${boxes[i - 1].id} overlaps ${boxes[i].id}`,
    )
  }
  const article = await page.locator("article.sidenotes").boundingBox()
  assert(article.y + article.height >= boxes.at(-1).bottom - 1, "article contains its last note")
  assert(
    await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth),
    "no page overflow",
  )
  if (width >= 960) {
    assert(
      boxes.every((b) => b.left >= article.x + article.width),
      "notes stay outside prose",
    )
  } else {
    assert(
      boxes.every((b) => b.left >= article.x - 1 && b.right <= article.x + article.width + 1),
      "notes become inline",
    )
  }
}

async function main() {
  const server = http.createServer((req, res) => void serve(req, res, { public: output }))
  await new Promise((resolve) => server.listen(0, "127.0.0.1", resolve))
  const url = `http://127.0.0.1:${server.address().port}`
  let browser
  try {
    browser = await chromium.launch()
    const page = await browser.newPage({ reducedMotion: "reduce" })
    await page.goto(url)
    await page.evaluate(() => document.fonts.ready)
    for (const width of [390, 959, 960, 1440, 1920]) {
      await page.setViewportSize({ width, height: 1000 })
      for (const theme of ["light", "dark"]) {
        if ((await page.locator("html").getAttribute("saved-theme")) !== theme)
          await page.locator(".darkmode").click()
        for (const open of [false, true]) {
          await page.locator("#extra").evaluate((el, value) => {
            el.open = value
          }, open)
          await checkLayout(page, width)
        }
        await page.locator('[aria-describedby="long-note"]').focus()
        assert(await page.locator("#long-note").isVisible())
        const height = await page.locator("article.sidenotes").evaluate((el) => el.offsetHeight)
        await page.getByRole("button", { name: "Media preview" }).focus()
        const media = page.locator(".sidenote--hover-media")
        assert.equal(await media.evaluate((el) => getComputedStyle(el).position), "absolute")
        assert.equal(await media.evaluate((el) => getComputedStyle(el).opacity), "1")
        assert.equal(
          await page.locator("article.sidenotes").evaluate((el) => el.offsetHeight),
          height,
        )
        await page.keyboard.press("Escape")
        assert.equal(await media.evaluate((el) => getComputedStyle(el).visibility), "hidden")
        console.log(
          `PASS ${width}px ${theme}: stacked/inline notes, disclosure resize, media reveal`,
        )
      }
    }
    await page.setViewportSize({ width: 1440, height: 1000 })
    await page.locator("article").getByRole("link", { name: "Another page", exact: true }).click()
    await page.locator("article").getByRole("link", { name: "Back", exact: true }).click()
    await page.locator("#long-note").waitFor()
    await checkLayout(page, 1440)
    await page.screenshot({ path: path.join(work, "sidenotes-desktop.png"), fullPage: true })
    console.log(`PASS SPA navigation; artifacts: ${work}`)
  } finally {
    if (browser) await browser.close()
    await new Promise((resolve) => server.close(resolve))
  }
}

main().catch((error) => {
  console.error(error)
  process.exitCode = 1
})
