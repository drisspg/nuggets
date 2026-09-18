import assert from "node:assert/strict"
import { test } from "node:test"
import { numericDomain, parseChart } from "./chart"

const series = {
  name: "observations",
  mode: "line",
  points: [
    { x: 1, y: 2 },
    { x: 3, y: 4 },
  ],
}
const chart = (overrides: Record<string, unknown> = {}) => ({
  version: 1,
  x: { label: "Step" },
  y: { label: "Loss (lower is better)" },
  series: [series],
  ...overrides,
})
const withSeries = (overrides: Record<string, unknown>) =>
  chart({ series: [{ ...series, ...overrides }] })
const withPoint = (point: unknown) => withSeries({ points: [point] })

function freeze(value: unknown) {
  if (value && typeof value === "object") {
    Object.freeze(value)
    for (const child of Object.values(value)) freeze(child)
  }
}

test("parseChart preserves exact values, gaps, styling, metadata, and input ownership", () => {
  const input = {
    ...chart(),
    intervalLabel: "Estimate ± logged uncertainty",
    x: { label: "Step", format: "number", includeZero: false, domain: [0, 10] },
    y: { label: "Loss", format: "scientific" },
    series: [
      {
        name: "A",
        mode: "line",
        color: "amber",
        dash: "dash",
        marker: "diamond",
        points: [
          {
            x: 1,
            y: 0.12345678901234566,
            yLow: 0.12,
            yHigh: 0.13,
            details: { se: 0.0001, label: "<raw>", empty: "" },
          },
          { x: 2, y: null },
          { x: 3, y: 0.11 },
        ],
      },
    ],
  }
  freeze(input)
  const result = parseChart(input)
  assert.deepEqual(result, input)
  assert.notEqual(result.x, input.x)
  assert.notEqual(result.series, input.series)
  result.x.domain![0] = -1
  result.series[0].points[0].details!.label = "changed"
  assert.equal(input.x.domain[0], 0)
  assert.equal(input.series[0].points[0].details!.label, "<raw>")
  assert.equal(result.series[0].points[1].y, null)
})

test("category charts preserve order and allow unsorted/repeated x for points", () => {
  const input = chart({
    y: { label: "Run", categories: ["seed 42", "seed 11"] },
    intervalLabel: "Estimate ± 1.96 SE",
    series: [
      {
        name: "runs",
        mode: "points",
        color: "purple",
        marker: "diamond",
        points: [
          { x: 2, y: "seed 42", xLow: 1, xHigh: 3 },
          { x: 1, y: "seed 11" },
          { x: 1, y: "seed 42" },
        ],
      },
    ],
  })
  const result = parseChart(input)
  assert.deepEqual(result, input)
  assert.notEqual(result.y, input.y)
  assert.throws(() => numericDomain(result, "y"), /chart\.y\.domain: category axes/)
})

test("reject malformed schema and unsupported features with field locations", async (t) => {
  const cases: [string, unknown, RegExp][] = [
    ["null chart", null, /chart: expected a plain object/],
    ["array chart", [], /chart: expected a plain object/],
    ["class chart", new Date(), /chart: expected a plain object/],
    ["wrong version", chart({ version: 2 }), /chart\.version/],
    ["string version", chart({ version: "1" }), /chart\.version/],
    ["unknown root key", chart({ zoom: true }), /chart\.zoom: unknown field/],
    ["log scale", chart({ x: { label: "x", scale: "log" } }), /chart\.x\.scale: unknown field/],
    ["category x", chart({ x: { label: "x", categories: ["a"] } }), /chart\.x\.categories/],
    ["missing axis", chart({ x: undefined }), /chart\.x/],
    ["empty label", chart({ y: { label: " " } }), /chart\.y\.label/],
    ["invalid format", chart({ x: { label: "x", format: "percent" } }), /chart\.x\.format/],
    ["string boolean", chart({ x: { label: "x", includeZero: "true" } }), /chart\.x\.includeZero/],
    ["null optional format", chart({ x: { label: "x", format: null } }), /chart\.x\.format/],
    ["empty series", chart({ series: [] }), /chart\.series: expected a nonempty array/],
    ["missing series", chart({ series: undefined }), /chart\.series/],
    [
      "sparse series",
      chart({ series: new Array(1) }),
      /chart\.series\[0\]: expected a plain object/,
    ],
    ["empty name", withSeries({ name: "" }), /chart\.series\[0\]\.name/],
    ["duplicate names", chart({ series: [series, series] }), /chart\.series\[1\]\.name: duplicate/],
    ["unknown mode", withSeries({ mode: "bar" }), /chart\.series\[0\]\.mode/],
    ["unknown series key", withSeries({ smooth: true }), /chart\.series\[0\]\.smooth/],
    ["unknown color", withSeries({ color: "#f00" }), /chart\.series\[0\]\.color/],
    ["unknown dash", withSeries({ dash: "dots" }), /chart\.series\[0\]\.dash/],
    ["unknown marker", withSeries({ marker: "square" }), /chart\.series\[0\]\.marker/],
    ["empty points", withSeries({ points: [] }), /chart\.series\[0\]\.points/],
    [
      "sparse points",
      withSeries({ points: [{ x: 1, y: 2 }, , { x: 3, y: 4 }] }),
      /points\[1\]: expected a plain object/,
    ],
    [
      "no plottable points",
      withPoint({ x: 1, y: null }),
      /chart\.series\[0\]\.points: series must contain a plottable/,
    ],
    ["unknown point key", withPoint({ x: 1, y: 2, z: 3 }), /points\[0\]\.z: unknown field/],
    ["missing y", withPoint({ x: 1 }), /points\[0\]\.y: expected a finite number/],
    [
      "unsorted x",
      withSeries({
        points: [
          { x: 2, y: 1 },
          { x: 1, y: 2 },
        ],
      }),
      /points\[1\]\.x: line points/,
    ],
    [
      "duplicate x across a gap",
      withSeries({
        points: [
          { x: 1, y: 1 },
          { x: 1, y: null },
        ],
      }),
      /points\[1\]\.x: line points/,
    ],
    [
      "array details",
      withPoint({ x: 1, y: 2, details: [] }),
      /points\[0\]\.details: expected a plain object/,
    ],
    ["nested details", withPoint({ x: 1, y: 2, details: { nested: {} } }), /details\.nested/],
    ["boolean details", withPoint({ x: 1, y: 2, details: { flag: true } }), /details\.flag/],
    ["nonfinite details", withPoint({ x: 1, y: 2, details: { se: Infinity } }), /details\.se/],
  ]
  for (const [name, input, error] of cases) {
    await t.test(name, () => assert.throws(() => parseChart(input), error))
  }
})

test("numeric fields reject strings, null, and nonfinite numbers rather than coercing", async (t) => {
  for (const field of ["x", "y", "xLow", "xHigh", "yLow", "yHigh"]) {
    for (const value of [
      "1",
      NaN,
      Infinity,
      -Infinity,
      undefined,
      ...(field === "y" ? [] : [null]),
    ]) {
      await t.test(`${field} = ${String(value)}`, () => {
        const point = { x: 1, y: 2, xLow: 0, xHigh: 2, yLow: 1, yHigh: 3, [field]: value }
        const input = { ...withPoint(point), intervalLabel: "Logged bounds" }
        assert.throws(
          () => parseChart(input),
          new RegExp(`points\\[0\\]\\.${field}: expected a finite number`),
        )
      })
    }
  }
})

test("intervals require a pair, a numeric estimate, ordered enclosing bounds, and a label", async (t) => {
  const cases: [string, unknown, RegExp][] = [
    ["missing xHigh", { x: 1, y: 2, xLow: 0 }, /xLow and xHigh must be provided together/],
    ["missing xLow", { x: 1, y: 2, xHigh: 3 }, /xLow and xHigh must be provided together/],
    ["missing yHigh", { x: 1, y: 2, yLow: 0 }, /yLow and yHigh must be provided together/],
    ["missing yLow", { x: 1, y: 2, yHigh: 3 }, /yLow and yHigh must be provided together/],
    [
      "reversed bounds",
      { x: 1, y: 2, yLow: 3, yHigh: 1 },
      /yLow: bounds must be ordered and enclose/,
    ],
    [
      "estimate below bounds",
      { x: 1, y: 2, xLow: 2, xHigh: 3 },
      /xLow: bounds must be ordered and enclose/,
    ],
    [
      "estimate above bounds",
      { x: 1, y: 2, yLow: 0, yHigh: 1 },
      /yLow: bounds must be ordered and enclose/,
    ],
    [
      "bounds on gap",
      { x: 1, y: null, yLow: 0, yHigh: 1 },
      /yLow: bounds require a numeric estimate/,
    ],
    [
      "x bounds on gap",
      { x: 1, y: null, xLow: 0, xHigh: 2 },
      /xLow: bounds require a numeric estimate/,
    ],
    ["missing label", { x: 1, y: 2, yLow: 1, yHigh: 3 }, /chart\.intervalLabel: required/],
  ]
  for (const [name, point, error] of cases) {
    await t.test(name, () => assert.throws(() => parseChart(withPoint(point)), error))
  }
  assert.throws(() => parseChart(chart({ intervalLabel: " " })), /chart\.intervalLabel/)
  const input = {
    ...withPoint({ x: 1, y: 2, xLow: 1, xHigh: 1, yLow: 2, yHigh: 2 }),
    intervalLabel: "Exact",
  }
  assert.deepEqual(parseChart(input), input)
})

test("categories reject duplicates, unknown labels, nonstrings, and numeric-axis options", async (t) => {
  const categories = { label: "Run", categories: ["a", "b"] }
  const cases: [string, unknown, RegExp][] = [
    ["empty categories", { ...categories, categories: [] }, /chart\.y\.categories/],
    [
      "duplicate categories",
      { ...categories, categories: ["a", "a"] },
      /chart\.y\.categories\[1\]: duplicate/,
    ],
    ["numeric category", { ...categories, categories: [1] }, /chart\.y\.categories\[0\]/],
    ["blank category", { ...categories, categories: [" "] }, /chart\.y\.categories\[0\]/],
    [
      "domain on category axis",
      { ...categories, domain: [0, 2] },
      /chart\.y\.domain: unknown field/,
    ],
  ]
  for (const [name, y, error] of cases) {
    await t.test(name, () => assert.throws(() => parseChart(chart({ y })), error))
  }
  for (const y of ["missing", 1, null]) {
    assert.throws(
      () => parseChart({ ...withSeries({ mode: "points", points: [{ x: 1, y }] }), y: categories }),
      /points\[0\]\.y/,
    )
  }
  assert.throws(
    () => parseChart(chart({ y: categories })),
    /series\[0\]\.mode: category charts require points mode/,
  )
  assert.throws(
    () =>
      parseChart({
        ...withSeries({ mode: "points", points: [{ x: 1, y: "a", yLow: 0, yHigh: 2 }] }),
        y: categories,
        intervalLabel: "Bounds",
      }),
    /yLow: bounds require a numeric estimate/,
  )
})

test("automatic domains include every observation and interval, with 5% padding", () => {
  const input = chart({
    intervalLabel: "Bounds",
    series: [
      {
        name: "A",
        mode: "line",
        points: [
          { x: 10, y: 0, xLow: 8, xHigh: 12 },
          { x: 20, y: null },
        ],
      },
      {
        name: "B",
        mode: "points",
        points: [{ x: 40, y: 6, xLow: 35, xHigh: 44, yLow: -4, yHigh: 10 }],
      },
    ],
  })
  const parsed = parseChart(input)
  assert.deepEqual(numericDomain(parsed, "x"), [6.2, 45.8])
  assert.deepEqual(numericDomain(parsed, "y"), [-4.7, 10.7])
  const gap = parseChart(
    withSeries({
      points: [
        { x: 1, y: 10 },
        { x: 100, y: null },
      ],
    }),
  )
  assert.deepEqual(numericDomain(gap, "x"), [-3.95, 104.95])
  assert.deepEqual(numericDomain(gap, "y"), [9.5, 10.5])
})

test("zero inclusion is opt-in and constant/subnormal domains remain finite and increasing", () => {
  const input = withSeries({
    points: [
      { x: 1, y: 10 },
      { x: 2, y: 20 },
    ],
  })
  assert.deepEqual(numericDomain(parseChart(input), "y"), [9.5, 20.5])
  assert.deepEqual(
    numericDomain(parseChart({ ...input, y: { label: "y", includeZero: true } }), "y"),
    [-1, 21],
  )
  assert.deepEqual(numericDomain(parseChart(withPoint({ x: 20, y: 0 })), "x"), [19, 21])
  assert.deepEqual(numericDomain(parseChart(withPoint({ x: 20, y: 0 })), "y"), [-0.05, 0.05])
  assert.deepEqual(numericDomain(parseChart(withPoint({ x: Number.MIN_VALUE, y: -20 })), "x"), [
    0,
    2 * Number.MIN_VALUE,
  ])
  assert.deepEqual(numericDomain(parseChart(withPoint({ x: 1, y: -20 })), "y"), [-21, -19])
})

test("explicit domains are unpadded, increasing finite pairs encompassing data, bounds, and requested zero", async (t) => {
  const parsed = parseChart(chart({ x: { label: "x", domain: [1, 3] } }))
  const result = numericDomain(parsed, "x")
  assert.deepEqual(result, [1, 3])
  assert.notEqual(result, parsed.x.domain)
  for (const domain of [
    [0],
    [0, 3, 4],
    [3, 1],
    [1, 1],
    ["0", 3],
    [0, Infinity],
    [NaN, 3],
    [2, 3],
    [1, 2],
  ]) {
    await t.test(JSON.stringify(domain), () =>
      assert.throws(() => parseChart(chart({ x: { label: "x", domain } })), /chart\.x\.domain/),
    )
  }
  assert.throws(
    () => parseChart(chart({ x: { label: "x", domain: [1, 3], includeZero: true } })),
    /chart\.x\.domain: must encompass/,
  )
  assert.throws(
    () =>
      parseChart({
        ...withPoint({ x: 1, y: 2, yLow: -1, yHigh: 3 }),
        intervalLabel: "Bounds",
        y: { label: "y", domain: [0, 4] },
      }),
    /chart\.y\.domain: must encompass/,
  )
})

test("reject data-range and padding overflow rather than emitting Infinity or NaN", () => {
  const max = Number.MAX_VALUE
  assert.throws(
    () =>
      parseChart(
        withSeries({
          points: [
            { x: -max, y: 1 },
            { x: max, y: 2 },
          ],
        }),
      ),
    /chart\.x\.domain: range overflows/,
  )
  assert.throws(
    () => parseChart(withPoint({ x: 1, y: max })),
    /chart\.y\.domain: padded range overflows/,
  )
  assert.throws(
    () => parseChart(chart({ x: { label: "x", domain: [-max, max] } })),
    /chart\.x\.domain: range overflows/,
  )
})
