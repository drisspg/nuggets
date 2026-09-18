const assert = require("node:assert/strict")

/** Exercise viewport gestures without coupling the test to tick rounding or series colors. */
module.exports = async function checkChartViewport(page, chart, categorical = false) {
  const svg = chart.locator(".native-chart-plot > svg")
  const ranges = () =>
    svg.evaluate((node) => ({
      x: JSON.parse(node.dataset.xDomain),
      y: JSON.parse(node.dataset.yDomain),
    }))
  const points = () =>
    chart
      .locator(".native-chart-point")
      .evaluateAll((nodes) => nodes.map((node) => [node.dataset.x, node.dataset.y]))
  const size = (range) => range[1] - range[0]
  const close = (a, b) => Math.abs(a - b) <= Math.max(Math.abs(a), Math.abs(b), 1e-12) * 1e-10
  const geometry = () =>
    svg.evaluate((node) => {
      const box = node.getBoundingClientRect(),
        view = node.viewBox.baseVal
      const clip = node.querySelector("clipPath rect")
      const sx = box.width / view.width,
        sy = box.height / view.height
      return {
        x: box.x + Number(clip.getAttribute("x")) * sx,
        y: box.y + Number(clip.getAttribute("y")) * sy,
        width: Number(clip.getAttribute("width")) * sx,
        height: Number(clip.getAttribute("height")) * sy,
      }
    })
  const drag = async (from, to, shift = false) => {
    await svg.scrollIntoViewIfNeeded()
    const box = await geometry()
    if (shift) await page.keyboard.down("Shift")
    await page.mouse.move(box.x + box.width * from[0], box.y + box.height * from[1])
    await page.mouse.down()
    await page.mouse.move(box.x + box.width * to[0], box.y + box.height * to[1], { steps: 8 })
    await page.mouse.up()
    if (shift) await page.keyboard.up("Shift")
  }
  await svg.focus()
  await page.keyboard.press("0")
  await page.keyboard.press("Escape")
  const full = await ranges(),
    original = await points()
  assert.equal(await chart.locator(".native-chart-mode").count(), 0, "no text status badge")
  assert.equal(
    await chart.locator(".native-chart-pin-toggle").getAttribute("aria-pressed"),
    "false",
  )
  await chart.locator(".native-chart-pin-toggle").click()
  assert.equal(await chart.getAttribute("data-pinned"), "true")
  assert.notEqual(
    await chart
      .locator(".native-chart-pin-icon")
      .evaluate((node) => getComputedStyle(node).display),
    "none",
  )
  await chart.locator(".native-chart-pin-toggle").click()
  assert.equal(await chart.getAttribute("data-pinned"), "false")
  await drag([0.2, 0.2], [0.8, 0.8])
  const zoom = await ranges()
  // Browser pointer coordinates can be quantized to CSS pixels; the pure range tests are exact.
  assert(Math.abs(size(zoom.x) / size(full.x) - 0.6) < 0.015, JSON.stringify({ full, zoom }))
  assert(
    Math.abs(size(zoom.y) / size(full.y) - (categorical ? 1 : 0.6)) < 0.015,
    JSON.stringify({ full, zoom }),
  )
  assert.equal(await chart.getAttribute("data-zoomed"), "true")
  assert.equal(
    await chart.getAttribute("data-pinned"),
    "false",
    "drag must not end as a click-to-pin",
  )
  assert(
    (await chart.locator(".native-chart-data-layer").getAttribute("clip-path")).startsWith(
      "url(#native-chart-clip-",
    ),
  )
  assert.deepEqual(await points(), original, "zoom does not modify observations")
  assert(
    await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth),
    "clipped data must not create page overflow",
  )
  await drag([0.5, 0.5], [0.6, 0.6], true)
  const pan = await ranges()
  assert(close(size(pan.x), size(zoom.x)))
  assert(pan.x[0] < zoom.x[0])
  if (!categorical) assert(pan.y[0] > zoom.y[0])
  const viewport = page.viewportSize()
  await page.setViewportSize({ ...viewport, width: viewport.width === 390 ? 960 : 390 })
  await page.waitForFunction(
    (node) =>
      Number(node.getAttribute("viewBox").split(" ")[2]) ===
      Math.floor(node.getBoundingClientRect().width),
    await svg.elementHandle(),
  )
  assert.deepEqual(await ranges(), pan, "resize preserves the selected window")
  await page.setViewportSize(viewport)
  await page.locator(".darkmode").click()
  assert.deepEqual(await ranges(), pan, "theme changes preserve the selected window")
  await page.locator(".darkmode").click()
  await chart.locator(".native-chart-reset").click()
  assert.deepEqual(await ranges(), full)
  assert.equal(await chart.getAttribute("data-zoomed"), "false")
  await svg.focus()
  await page.keyboard.press("+")
  assert(close(size((await ranges()).x), size(full.x) / 2))
  const keyboardZoom = await ranges()
  await page.keyboard.press("Shift+ArrowRight")
  assert((await ranges()).x[0] > keyboardZoom.x[0])
  await page.keyboard.press("-")
  assert.deepEqual(await ranges(), full)
  await page.keyboard.press("+")
  await svg.dblclick()
  assert.deepEqual(await ranges(), full, "double-click resets the viewport")
  // No points in this upper-right region; reset and pointer inspection must remain usable.
  await drag([0.85, 0.02], [0.98, 0.15])
  assert.equal(await chart.getAttribute("data-zoomed"), "true")
  assert.equal(
    await chart.locator(".native-chart-pin-toggle").getAttribute("aria-disabled"),
    "true",
  )
  assert(
    (await chart.locator(".native-chart-coordinate-value").innerText()).includes(
      "No point centers",
    ),
  )
  assert.equal(await chart.locator(".native-chart-zero").count(), 0)
  assert(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth))
  await chart.locator(".native-chart-reset").click()
  assert.deepEqual(await ranges(), full)
  await svg.scrollIntoViewIfNeeded()
  const box = await geometry()
  await page.mouse.move(box.x + box.width * 0.2, box.y + box.height * 0.2)
  await page.mouse.down()
  await page.mouse.move(box.x + box.width * 0.6, box.y + box.height * 0.6, { steps: 5 })
  const beforeEscape = await page.evaluate(() => scrollY)
  await page.keyboard.press("Escape")
  await page.mouse.up()
  assert.deepEqual(await ranges(), full)
  assert.equal(await chart.getAttribute("data-pinned"), "false")
  assert(Math.abs((await page.evaluate(() => scrollY)) - beforeEscape) < 1)
  assert.deepEqual(await points(), original)

  // A drag beyond an edge must use the same clamped extent as its preview.
  await svg.scrollIntoViewIfNeeded()
  let boundary = await geometry()
  await page.mouse.move(boundary.x + boundary.width - 3, boundary.y + boundary.height / 2)
  await page.mouse.down()
  await page.mouse.move(boundary.x + boundary.width + 20, boundary.y + boundary.height / 2, {
    steps: 4,
  })
  await page.mouse.up()
  assert.deepEqual(await ranges(), full, "edge-start drags must not create an unseen sliver zoom")
  assert.equal(await chart.getAttribute("data-pinned"), "false")
  assert((await chart.locator(".native-chart-status").textContent()).includes("unchanged"))

  boundary = await geometry()
  await page.mouse.move(boundary.x + boundary.width / 2, boundary.y + boundary.height / 2)
  await page.mouse.down()
  await page.mouse.move(
    boundary.x + boundary.width / 2 + (categorical ? 0 : 5),
    boundary.y + boundary.height / 2 + (categorical ? 40 : 5),
  )
  await page.mouse.up()
  assert.deepEqual(await ranges(), full)
  assert((await chart.locator(".native-chart-status").textContent()).includes("unchanged"))

  for (const cancel of ["reset", "resize", "pointercancel"]) {
    await svg.focus()
    await page.keyboard.press("+")
    const beforeDrag = await ranges()
    await svg.scrollIntoViewIfNeeded()
    boundary = await geometry()
    await page.keyboard.down("Shift")
    await page.mouse.move(boundary.x + boundary.width * 0.5, boundary.y + boundary.height * 0.5)
    await page.mouse.down()
    await page.mouse.move(boundary.x + boundary.width * 0.65, boundary.y + boundary.height * 0.65, {
      steps: 5,
    })
    await page.keyboard.up("Shift")
    if (cancel === "reset") await page.keyboard.press("0")
    else if (cancel === "resize") {
      const previousSize = page.viewportSize()
      await page.setViewportSize({ ...previousSize, width: previousSize.width === 390 ? 960 : 390 })
      await page.waitForFunction(
        (node) =>
          Number(node.getAttribute("viewBox").split(" ")[2]) ===
          Math.floor(node.getBoundingClientRect().width),
        await svg.elementHandle(),
      )
      await page.setViewportSize(previousSize)
    } else {
      assert(await svg.evaluate((node) => node.hasPointerCapture(1)))
      await svg.dispatchEvent("pointercancel", { pointerId: 1, pointerType: "mouse" })
    }
    await page.mouse.up()
    await page.keyboard.up("Shift")
    assert.equal(
      await chart.getAttribute("data-pinned"),
      "false",
      `${cancel} must suppress a trailing click`,
    )
    assert.deepEqual(
      await ranges(),
      cancel === "reset" ? full : beforeDrag,
      `${cancel} restores the intended viewport`,
    )
    await svg.focus()
    await page.keyboard.press("0")
  }
}
