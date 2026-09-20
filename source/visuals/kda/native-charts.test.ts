import assert from "node:assert/strict"
import { existsSync, readFileSync } from "node:fs"
import { resolve } from "node:path"
import { test } from "node:test"
import remarkParse from "remark-parse"
import { unified } from "unified"
import { visit } from "unist-util-visit"
import {
  parseChart,
  type ChartPoint,
  type ChartSeries,
  type ChartSpec,
} from "../../quartz/util/chart"
import {
  buildKdaCharts,
  CHART_NAMES,
  FINAL_CHECKPOINTS,
  loadMetrics,
  loadTrainingMetrics,
  METRICS_PATH,
  OUTPUT_DIR,
  type PairedRow,
} from "./native-charts"

const metrics = loadMetrics()
const training = loadTrainingMetrics()
const charts = buildKdaCharts(metrics, training)
const SCALED_STEPS = [1000, 2000, 3000, 4000, 5000, 6000, 7000, 7600]

const seriesByName = (spec: ChartSpec, name: string): ChartSeries => {
  const found = spec.series.find((s) => s.name === name)
  assert(found, `missing series ${name}`)
  return found
}

const numericY = (point: ChartPoint): number => {
  assert.equal(typeof point.y, "number")
  return point.y as number
}

test("builder is pure and stable", () => {
  const before = JSON.stringify([metrics, training])
  const again = buildKdaCharts(metrics, training)
  assert.equal(JSON.stringify([metrics, training]), before)
  assert.deepEqual(again, charts)
  assert.deepEqual(Object.keys(charts), [...CHART_NAMES])
  for (const name of CHART_NAMES) assert.equal(charts[name].version, 1)
})

test("training loss retains every logged step at full precision without evaluation intervals", () => {
  const spec = charts["training-loss"]
  assert.equal(spec.y.label, "Cross-entropy (lower is better)")
  assert.equal(spec.intervalLabel, undefined)
  assert.equal(spec.series.length, 2)
  for (const [index, key] of (["scaled_causal", "scaled_midpoint"] as const).entries()) {
    const rows = training.series[key].rows
    const series = spec.series[index]
    assert.equal(series.mode, "line")
    assert.equal(series.color, index === 0 ? "blue" : "amber")
    assert.equal(series.points.length, 7600)
    assert.deepEqual(
      series.points.map((point) => point.x),
      Array.from({ length: 7600 }, (_, i) => i + 1),
    )
    assert.deepEqual(
      series.points,
      rows.map(({ step, loss }) => ({ x: step, y: loss })),
    )
  }
  assert.equal(spec.series[0].points[0].y, 12.441706657409668)
  assert.equal(spec.series[0].points.at(-1)!.y, 3.1887118816375732)
  assert.equal(spec.series[1].points[0].y, 12.44172191619873)
  assert.equal(spec.series[1].points.at(-1)!.y, 3.1853528022766113)
})

test("scaled loss: four curves copy logged NLLs with the old color/dash distinctions", () => {
  const spec = charts["scaled-loss"]
  assert.equal(spec.y.label, "Cross-entropy (lower is better)")
  assert.equal(spec.intervalLabel, undefined)
  assert.equal(spec.series.length, 4)
  const expected = [
    ["Causal / parallel", "scaled_causal", "eval/all/parallel_nll", "blue", "solid"],
    ["Causal / AR", "scaled_causal", "eval/all/autoregressive_nll", "blue", "dash"],
    ["Midpoint / parallel", "scaled_midpoint", "eval/all/parallel_nll", "amber", "solid"],
    ["Midpoint / AR", "scaled_midpoint", "eval/all/autoregressive_nll", "amber", "dash"],
  ] as const
  expected.forEach(([name, key, field, color, dash], i) => {
    const series = spec.series[i]
    assert.equal(series.name, name)
    assert.equal(series.mode, "line")
    assert.equal(series.color, color)
    assert.equal(series.dash, dash)
    const rows = metrics.series[key].rows
    assert.equal(series.points.length, 8)
    assert.deepEqual(
      series.points,
      rows.map((row) => ({ x: row._step, y: row[field] })),
    )
    assert.deepEqual(
      series.points.map((p) => p.x),
      SCALED_STEPS,
    )
  })
  // Spot check exact snapshot values, not rounded.
  assert.equal(seriesByName(spec, "Causal / AR").points[7].y, 3.1945261961615827)
  assert.equal(seriesByName(spec, "Midpoint / parallel").points[0].y, 3.941782731419268)
})

test("scaled gap: two curves with one sequence-level standard error", () => {
  const spec = charts["scaled-gap"]
  assert.deepEqual(spec.y, {
    label: "AR − parallel",
    format: "scientific",
    includeZero: true,
  })
  assert.equal(spec.intervalLabel, "Mean gap ± 1 standard error")
  assert.equal(spec.series.length, 2)
  for (const [name, key, color] of [
    ["Causal reference", "scaled_causal", "blue"],
    ["Midpoint reference", "scaled_midpoint", "amber"],
  ] as const) {
    const series = seriesByName(spec, name)
    assert.equal(series.mode, "line")
    assert.equal(series.color, color)
    const rows = metrics.series[key].rows
    assert.equal(series.points.length, rows.length)
    rows.forEach((row, i) => {
      const point = series.points[i]
      const se = row["eval/all/gap_stderr_by_sequence"]
      assert.equal(point.x, row._step)
      assert.equal(point.y, row["eval/all/gap_nats"])
      const difference = row["eval/all/autoregressive_nll"] - row["eval/all/parallel_nll"]
      assert(Math.abs(numericY(point) - difference) < 1e-12, "logged gap matches NLL difference")
      assert.deepEqual(point.details, { "Sequence SE": se })
      assert.equal(point.yLow, row["eval/all/gap_nats"] - se)
      assert.equal(point.yHigh, row["eval/all/gap_nats"] + se)
      assert(point.yLow! <= numericY(point) && numericY(point) <= point.yHigh!)
    })
  }
})

test("paired checkpoints: 64-sequence line plus 1,024-sequence diamond points only", () => {
  const spec = charts["paired-checkpoints"]
  assert.deepEqual(spec.y, { label: "ΔG", format: "scientific", includeZero: true })
  assert.equal(spec.intervalLabel, "Mean paired difference ± 1 standard error")
  assert.equal(spec.series.length, 2)

  const sweep = seriesByName(spec, "64 sequences")
  assert.equal(sweep.mode, "line")
  assert.equal(sweep.color, "green")
  assert.equal(sweep.marker, "circle")
  assert.equal(sweep.points.length, 8)

  const big = seriesByName(spec, "1,024 sequences")
  assert.equal(big.mode, "points")
  assert.equal(big.color, "purple")
  assert.equal(big.marker, "diamond")
  assert.equal(big.dash, undefined)
  assert.equal(big.points.length, 2)
  assert.deepEqual(
    big.points.map((p) => p.x),
    [4000, 7600],
  )

  for (const [series, key] of [
    [sweep, "scaled_paired_64"],
    [big, "scaled_paired_1024"],
  ] as const) {
    const rows = metrics.series[key].rows
    rows.forEach((row, i) => {
      const point = series.points[i]
      const se = row["paired/gap_did_se"]
      assert.equal(point.x, row._step)
      assert.equal(point.y, row["paired/gap_did"])
      assert.deepEqual(point.details, { "Paired SE": se })
      assert.equal(point.yLow, row["paired/gap_did"] - se)
      assert.equal(point.yHigh, row["paired/gap_did"] + se)
    })
  }

  // Sweep DiD must equal midpoint gap minus causal gap at each aligned checkpoint.
  metrics.series.scaled_causal.rows.forEach((causal, i) => {
    const midpoint = metrics.series.scaled_midpoint.rows[i]
    const did = numericY(sweep.points[i])
    assert(Math.abs(did - (midpoint["eval/all/gap_nats"] - causal["eval/all/gap_nats"])) < 1e-12)
  })
})

test("independent final scaled DiD and interval bounds", () => {
  // Reference values computed outside the builder (decimal arithmetic) from metrics.json.
  const sweepFinal = seriesByName(charts["paired-checkpoints"], "64 sequences").points[7]
  assert.equal(sweepFinal.x, 7600)
  assert.equal(sweepFinal.y, -0.00030798983442430206)
  assert.equal(sweepFinal.details?.["Paired SE"], 0.00022530600260312265)
  assert(Math.abs(sweepFinal.yLow! - -0.00053329583702742471) < 1e-18)
  assert(Math.abs(sweepFinal.yHigh! - -0.00008268383182117941) < 1e-18)
  assert(sweepFinal.yHigh! < 0, "the one-SE bar need not include zero")

  const bigFinal = seriesByName(charts["paired-checkpoints"], "1,024 sequences").points[1]
  assert.equal(bigFinal.x, 7600)
  assert.equal(bigFinal.y, 8.819354240234556e-6)
  assert.equal(bigFinal.details?.["Paired SE"], 5.118582758300765e-5)
  assert(Math.abs(bigFinal.yLow! - -0.000042366473342773094) < 1e-18)
  assert(Math.abs(bigFinal.yHigh! - 0.000060005181823242206) < 1e-18)

  // Forest row for the 1.45B model is the same final 1,024-sequence checkpoint, on the x axis.
  const forest = charts["paired-seeds"].series[0].points[3]
  assert.equal(forest.y, "1.45B · seed 42")
  assert.equal(forest.x, bigFinal.y)
  assert.equal(forest.xLow, bigFinal.yLow)
  assert.equal(forest.xHigh, bigFinal.yHigh)
  assert.equal(forest.details?.["Training step"], 7600)
})

test("paired seeds: forest of final checkpoints with horizontal intervals in the old order", () => {
  const spec = charts["paired-seeds"]
  assert.equal(spec.intervalLabel, "Mean paired difference ± 1 standard error")
  assert.deepEqual(spec.x, { label: "ΔG", format: "scientific", includeZero: true })
  assert("categories" in spec.y)
  const categories = (spec.y as { categories: string[] }).categories
  assert.deepEqual(categories, [
    "520M · seed 42",
    "520M · seed 11",
    "520M · seed 23",
    "1.45B · seed 42",
  ])
  assert.equal(spec.series.length, 1)
  const series = spec.series[0]
  assert.equal(series.mode, "points")
  assert.equal(series.color, "purple")
  assert.equal(series.points.length, 4)
  assert.deepEqual(
    series.points.map((p) => p.y),
    categories,
  )
  FINAL_CHECKPOINTS.forEach(([key, label], i) => {
    const rows = metrics.series[key].rows as PairedRow[]
    const row = rows[rows.length - 1]
    const point = series.points[i]
    const se = row["paired/gap_did_se"]
    assert.equal(point.y, label)
    assert.equal(point.x, row["paired/gap_did"])
    assert.equal(point.xLow, row["paired/gap_did"] - se)
    assert.equal(point.xHigh, row["paired/gap_did"] + se)
    assert.equal(point.yLow, undefined)
    assert.equal(point.yHigh, undefined)
    assert.deepEqual(point.details, { "Training step": row._step, "Paired SE": se })
  })
  assert.deepEqual(
    series.points.map((p) => p.details?.["Training step"]),
    [4000, 4000, 4000, 7600],
  )
  assert.equal(series.points[0].x, -3.935295832625113e-5)
  assert.equal(series.points[1].x, 6.475663717455532e-5)
  assert.equal(series.points[2].x, -3.156410532945131e-6)
})

test("every numeric value is finite and every interval encloses its estimate", () => {
  for (const name of CHART_NAMES) {
    for (const series of charts[name].series) {
      assert(series.points.length > 0)
      for (const point of series.points) {
        assert(Number.isFinite(point.x))
        if (typeof point.y === "number") assert(Number.isFinite(point.y))
        for (const key of ["xLow", "xHigh", "yLow", "yHigh"] as const) {
          if (point[key] !== undefined) assert(Number.isFinite(point[key]))
        }
        if (point.xLow !== undefined) assert(point.xLow <= point.x && point.x <= point.xHigh!)
        if (point.yLow !== undefined) {
          const y = numericY(point)
          assert(point.yLow <= y && y <= point.yHigh!)
        }
        for (const value of Object.values(point.details ?? {})) {
          assert(typeof value === "string" || Number.isFinite(value))
        }
      }
      if (series.points.some((p) => p.xLow !== undefined || p.yLow !== undefined)) {
        assert(charts[name].intervalLabel, `${name} has intervals but no intervalLabel`)
      }
    }
  }
})

test("generated JSON assets match the builder output exactly", () => {
  for (const name of CHART_NAMES) {
    const path = resolve(OUTPUT_DIR, `${name}.json`)
    assert(existsSync(path), `${path} missing; run: npx tsx visuals/kda/native-charts.ts`)
    const onDisk = JSON.parse(readFileSync(path, "utf8"))
    assert.deepEqual(onDisk, JSON.parse(JSON.stringify(charts[name])), `${name}.json is stale`)
  }
  assert(existsSync(METRICS_PATH))
})

test("the KDA article uses native embeds for measured data and the analytic range chart", () => {
  const markdown = readFileSync(resolve(OUTPUT_DIR, "../../KDA Future Token Leakage.md"), "utf8")
  const tree = unified().use(remarkParse).parse(markdown)
  const embeds: Array<{ language: string; src: string }> = []
  visit(tree, "code", (node) => {
    if (node.lang === "chart" || node.lang === "plotly") {
      embeds.push({ language: node.lang, src: JSON.parse(node.value).src })
    }
  })
  for (const name of [...CHART_NAMES, "rebase-range"]) {
    assert.equal(
      embeds.filter((embed) => embed.language === "chart" && embed.src === `media/kda/${name}.json`)
        .length,
      1,
      `${name} must use the native chart embed`,
    )
    assert(
      !embeds.some((embed) => embed.src === `media/kda/${name}.html`),
      `${name} still uses the old iframe export`,
    )
  }
})

test("builder rejects misaligned checkpoints", () => {
  const broken = JSON.parse(JSON.stringify(metrics)) as typeof metrics
  broken.series.scaled_paired_64.rows.pop()
  assert.throws(() => buildKdaCharts(broken, training), /scaled_paired_64 checkpoints/)
})

test("builder rejects invalid logged standard errors without coercion", () => {
  for (const value of [-0.1, NaN, Infinity, "0.001"]) {
    const broken = structuredClone(metrics)
    broken.series.scaled_paired_1024.rows[0]["paired/gap_did_se"] = value as number
    assert.throws(() => buildKdaCharts(broken, training), /finite, nonnegative logged SE/)
  }
})

test("generated specs pass the shared parseChart contract", () => {
  for (const name of CHART_NAMES) {
    const raw = JSON.parse(readFileSync(resolve(OUTPUT_DIR, `${name}.json`), "utf8"))
    const parsed = parseChart(raw)
    assert.deepEqual(parsed, JSON.parse(JSON.stringify(charts[name])), name)
  }
})
