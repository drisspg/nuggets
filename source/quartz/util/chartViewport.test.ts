import assert from "node:assert/strict"
import test from "node:test"
import { panChartRange, scaleChartRange, selectChartRange, type ChartRange } from "./chartViewport"

test("box selection accepts both directions, clamps outside drags and rejects empty/nonfinite ranges", () => {
  const full: ChartRange = [0, 100]
  assert.deepEqual(selectChartRange(full, full, 0.2, 0.8), [20, 80])
  assert.deepEqual(selectChartRange(full, full, 0.8, 0.2), [20, 80])
  assert.deepEqual(selectChartRange(full, full, -2, 3), full)
  assert.equal(selectChartRange(full, full, 0.5, 0.5), null)
  assert.equal(selectChartRange(full, full, NaN, 1), null)
  assert.deepEqual(selectChartRange([20, 80], full, 0.25, 0.75), [35, 65])
})

test("pan preserves interval width and clamps at both full-data boundaries", () => {
  const full: ChartRange = [-100, 100]
  assert.deepEqual(panChartRange([-20, 20], full, 0.5), [0, 40])
  assert.deepEqual(panChartRange([-20, 20], full, -100), [-100, -60])
  assert.deepEqual(panChartRange([-20, 20], full, 100), [60, 100])
  assert.deepEqual(panChartRange(full, full, 1), full)
  assert.deepEqual(panChartRange([-20, 20], full, NaN), [-20, 20])
})

test("keyboard zoom respects anchors, minimum extent, full extent and invalid input", () => {
  const full: ChartRange = [0, 100]
  assert.deepEqual(scaleChartRange(full, full, 0.5), [25, 75])
  assert.deepEqual(scaleChartRange(full, full, 0.5, 0), [0, 50])
  assert.deepEqual(scaleChartRange(full, full, 0.5, 1), [50, 100])
  assert.deepEqual(scaleChartRange([25, 75], full, 2), full)
  assert.deepEqual(scaleChartRange([25, 75], full, Number.MAX_VALUE), full)
  assert.deepEqual(scaleChartRange(full, full, -1), full)
  const tight = scaleChartRange(full, full, 1e-30)
  assert(tight[1] > tight[0])
  assert(Math.abs(tight[1] - tight[0] - 0.0001) < 1e-10)
})

test("unrepresentable tiny windows fall back safely and undersized views respect the zoom limit", () => {
  const full: ChartRange = [1e15, 1e15 + 1]
  assert.equal(selectChartRange(full, full, 0.5, 0.50000001), null)
  const current: ChartRange = [1e15 + 0.5, 1e15 + 0.625]
  assert.deepEqual(
    scaleChartRange(current, full, 0.001),
    current,
    "precision limits must not reset an existing zoom",
  )
  const expanded = panChartRange([0.1, 0.10000000001], [0, 1], 0)
  assert(expanded[1] - expanded[0] >= 0.999999e-6)
  assert(expanded[0] >= 0 && expanded[1] <= 1)
})

test("scientific ranges and large finite values stay finite without mutating the source", () => {
  for (const full of [
    [-1e-4, 2e-4],
    [1e307, 1.1e307],
  ] as ChartRange[]) {
    const before = [...full]
    const view = selectChartRange(full, full, 0.2, 0.8)!
    for (const result of [view, panChartRange(view, full, 100), scaleChartRange(view, full, 0.1)]) {
      assert(result.every(Number.isFinite))
      assert(result[0] >= full[0] && result[1] <= full[1])
      assert(result[0] < result[1])
    }
    assert.deepEqual(full, before)
  }
})
