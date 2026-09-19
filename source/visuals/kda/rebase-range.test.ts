import assert from "node:assert/strict"
import { readFile } from "node:fs/promises"
import { test } from "node:test"
import { parseChart } from "../../quartz/util/chart"
import { buildRebaseRange, GATE_MAGNITUDE, RANGE_PATH } from "./rebase-range"

test("analytic range chart is valid and its committed asset is reproducible", async () => {
  const chart = buildRebaseRange()
  assert.doesNotThrow(() => parseChart(chart))
  assert.deepEqual(JSON.parse(await readFile(RANGE_PATH, "utf8")), chart)
  assert.deepEqual(
    chart.series.map((series) => [series.name, series.points.length]),
    [
      ["Inverse factor", 64],
      ["Decay factor", 64],
    ],
  )
})

test("FP32 limits are reference barriers, not legend series", () => {
  const chart = buildRebaseRange()
  assert.deepEqual(chart.references, [
    { y: 128, label: "FP32 overflow (+128)", color: "red" },
    { y: -126, label: "FP32 subnormal (−126)", color: "purple" },
  ])
  // The barriers are drawn from the reference list only: no constant series duplicates them.
  for (const series of chart.series) {
    assert(series.points.some((point) => point.y !== 128 && point.y !== -126))
  }
  assert.deepEqual(chart.y, { label: "Gate-factor exponent (base 2)", domain: [-480, 480] })
})

test("first-row factors accumulate over window minus one steps", () => {
  const [inverse, decay] = buildRebaseRange().series
  for (const window of [1, 16, 18, 19, 32, 64]) {
    const positive = inverse.points[window - 1]
    const negative = decay.points[window - 1]
    assert.equal(positive.x, window)
    assert.equal(positive.y, (window - 1) * GATE_MAGNITUDE)
    assert.equal(Number(positive.y) + Number(negative.y), 0)
    assert.equal(positive.details!["Steps from first row"], window - 1)
  }
})

test("FP32 overflows at eighteen steps, not at the sixteen-token tile", () => {
  assert(Number.isFinite(Math.fround(2 ** (15 * GATE_MAGNITUDE))))
  assert(Number.isFinite(Math.fround(2 ** (17 * GATE_MAGNITUDE))))
  assert.equal(Math.fround(2 ** (18 * GATE_MAGNITUDE)), Infinity)
  assert.equal(Math.fround(2 ** (31 * GATE_MAGNITUDE)), Infinity)
  // Crossing the normal floor does not immediately mean zero.
  const small = Math.fround(2 ** (-18 * GATE_MAGNITUDE))
  assert(small > 0 && small < 2 ** -126)
  assert.equal(Math.fround(2 ** (-21 * GATE_MAGNITUDE)), 0)
  const [inverse, decay] = buildRebaseRange().series
  assert.equal(inverse.points[18].details!["FP32 cast"], "Overflow")
  assert.equal(decay.points[18].details!["FP32 cast"], small)
  assert.equal(decay.points[21].details!["FP32 cast"], 0)
})
