/**
 * Build native chart specs for the KDA article from the metadata-free W&B snapshot.
 *
 * Run from `source/`: npx tsx visuals/kda/native-charts.ts
 * Writes content/media/kda/{scaled-loss,scaled-gap,paired-checkpoints,paired-seeds}.json.
 *
 * Values are copied from metrics.json at full precision. Intervals are the estimate ± 1.96 ×
 * the logged standard error (sequence-level for arm gaps, paired for differences of gaps); the
 * logged SE is kept in each point's details. Nothing is smoothed, rescaled, or recomputed.
 */

import { mkdirSync, readFileSync, writeFileSync } from "node:fs"
import { dirname, resolve } from "node:path"
import { fileURLToPath } from "node:url"
import { format, resolveConfig } from "prettier"
import {
  parseChart,
  type ChartPoint,
  type ChartSeries,
  type ChartSpec,
} from "../../quartz/util/chart"

export const Z_95 = 1.96

export interface ArmRow {
  _step: number
  "eval/all/autoregressive_nll": number
  "eval/all/parallel_nll": number
  "eval/all/gap_nats": number
  "eval/all/gap_stderr_by_sequence": number
}

export interface PairedRow {
  _step: number
  "paired/gap_did": number
  "paired/gap_did_se": number
}

export interface KdaMetrics {
  description: string
  series: {
    scaled_causal: { num_sequences: number; seed: number; rows: ArmRow[] }
    scaled_midpoint: { num_sequences: number; seed: number; rows: ArmRow[] }
    scaled_paired_64: { num_sequences: number; seed: number; rows: PairedRow[] }
    scaled_paired_1024: { num_sequences: number; seed: number; rows: PairedRow[] }
    pilot_seed42: { num_sequences: number; seed: number; rows: PairedRow[] }
    pilot_seed11: { num_sequences: number; seed: number; rows: PairedRow[] }
    pilot_seed23: { num_sequences: number; seed: number; rows: PairedRow[] }
  }
}

export const CHART_NAMES = [
  "scaled-loss",
  "scaled-gap",
  "paired-checkpoints",
  "paired-seeds",
] as const
export type ChartName = (typeof CHART_NAMES)[number]

/** Forest rows, top to bottom, as (series key, category label). */
export const FINAL_CHECKPOINTS: ReadonlyArray<readonly [keyof KdaMetrics["series"], string]> = [
  ["pilot_seed42", "520M · seed 42"],
  ["pilot_seed11", "520M · seed 11"],
  ["pilot_seed23", "520M · seed 23"],
  ["scaled_paired_1024", "1.45B · seed 42"],
]

const STEP_AXIS = { label: "Training step" } as const
const SEQUENCE_INTERVAL = "estimate ± 1.96 × logged sequence-level SE (approximate pointwise 95%)"
const PAIRED_INTERVAL = "estimate ± 1.96 × logged paired SE (approximate pointwise 95%)"

const KDA_DIR = dirname(fileURLToPath(import.meta.url))
export const METRICS_PATH = resolve(KDA_DIR, "metrics.json")
export const OUTPUT_DIR = resolve(KDA_DIR, "../../content/media/kda")

export function loadMetrics(path = METRICS_PATH): KdaMetrics {
  return JSON.parse(readFileSync(path, "utf8")) as KdaMetrics
}

function interval(estimate: number, se: number): [number, number] {
  if (!Number.isFinite(estimate) || !Number.isFinite(se) || se < 0) {
    throw new Error("Intervals require a finite estimate and a finite, nonnegative logged SE")
  }
  return [estimate - Z_95 * se, estimate + Z_95 * se]
}

function lossSeries(
  rows: ArmRow[],
  arm: "Causal" | "Midpoint",
  color: ChartSeries["color"],
): ChartSeries[] {
  const modes = [
    ["parallel", "eval/all/parallel_nll", "solid"],
    ["AR", "eval/all/autoregressive_nll", "dash"],
  ] as const
  return modes.map(([label, key, dash]) => ({
    name: `${arm} / ${label}`,
    mode: "line",
    color,
    dash,
    marker: "circle",
    points: rows.map((row) => ({ x: row._step, y: row[key] })),
  }))
}

function gapSeries(
  rows: ArmRow[],
  arm: "Causal" | "Midpoint",
  color: ChartSeries["color"],
): ChartSeries {
  return {
    name: `${arm} reference`,
    mode: "line",
    color,
    dash: "solid",
    marker: "circle",
    points: rows.map((row) => {
      const se = row["eval/all/gap_stderr_by_sequence"]
      const [yLow, yHigh] = interval(row["eval/all/gap_nats"], se)
      return {
        x: row._step,
        y: row["eval/all/gap_nats"],
        yLow,
        yHigh,
        details: { "Sequence SE": se },
      }
    }),
  }
}

function pairedPoints(rows: PairedRow[]): ChartPoint[] {
  return rows.map((row) => {
    const se = row["paired/gap_did_se"]
    const [yLow, yHigh] = interval(row["paired/gap_did"], se)
    return { x: row._step, y: row["paired/gap_did"], yLow, yHigh, details: { "Paired SE": se } }
  })
}

function assertAlignedSteps(metrics: KdaMetrics): void {
  const { scaled_causal, scaled_midpoint, scaled_paired_64 } = metrics.series
  const steps = scaled_causal.rows.map((row) => row._step).join(",")
  for (const [name, series] of [
    ["scaled_midpoint", scaled_midpoint],
    ["scaled_paired_64", scaled_paired_64],
  ] as const) {
    const other = series.rows.map((row) => row._step).join(",")
    if (other !== steps) {
      throw new Error(`${name} checkpoints [${other}] do not match scaled_causal [${steps}]`)
    }
  }
}

/** Pure: builds the four article charts from the snapshot without touching the filesystem. */
export function buildKdaCharts(metrics: KdaMetrics): Record<ChartName, ChartSpec> {
  assertAlignedSteps(metrics)
  const { scaled_causal, scaled_midpoint, scaled_paired_64, scaled_paired_1024 } = metrics.series

  const scaledLoss: ChartSpec = {
    version: 1,
    x: STEP_AXIS,
    y: { label: "NLL (nats/token; lower is better)" },
    series: [
      ...lossSeries(scaled_causal.rows, "Causal", "blue"),
      ...lossSeries(scaled_midpoint.rows, "Midpoint", "amber"),
    ],
  }

  const scaledGap: ChartSpec = {
    version: 1,
    x: STEP_AXIS,
    y: { label: "AR − parallel (nats/token)", format: "scientific", includeZero: true },
    intervalLabel: SEQUENCE_INTERVAL,
    series: [
      gapSeries(scaled_causal.rows, "Causal", "blue"),
      gapSeries(scaled_midpoint.rows, "Midpoint", "amber"),
    ],
  }

  const pairedCheckpoints: ChartSpec = {
    version: 1,
    x: STEP_AXIS,
    y: { label: "ΔG (nats/token)", format: "scientific", includeZero: true },
    intervalLabel: PAIRED_INTERVAL,
    series: [
      {
        name: "64 sequences",
        mode: "line",
        color: "green",
        dash: "solid",
        marker: "circle",
        points: pairedPoints(scaled_paired_64.rows),
      },
      {
        name: "1,024 sequences",
        mode: "points",
        color: "purple",
        marker: "diamond",
        points: pairedPoints(scaled_paired_1024.rows),
      },
    ],
  }

  const pairedSeeds: ChartSpec = {
    version: 1,
    x: { label: "ΔG (nats/token)", format: "scientific", includeZero: true },
    y: { label: "Final checkpoint", categories: FINAL_CHECKPOINTS.map(([, label]) => label) },
    intervalLabel: PAIRED_INTERVAL,
    series: [
      {
        name: "Final checkpoint ΔG",
        mode: "points",
        color: "purple",
        marker: "circle",
        points: FINAL_CHECKPOINTS.map(([key, label]) => {
          const rows = metrics.series[key].rows as PairedRow[]
          const row = rows[rows.length - 1]
          const se = row["paired/gap_did_se"]
          const [xLow, xHigh] = interval(row["paired/gap_did"], se)
          return {
            x: row["paired/gap_did"],
            y: label,
            xLow,
            xHigh,
            details: { "Training step": row._step, "Paired SE": se },
          }
        }),
      },
    ],
  }

  return {
    "scaled-loss": scaledLoss,
    "scaled-gap": scaledGap,
    "paired-checkpoints": pairedCheckpoints,
    "paired-seeds": pairedSeeds,
  }
}

/** Writes the four specs as prettier-formatted JSON so `npm run check` stays clean. */
export async function writeKdaCharts(
  outputDir = OUTPUT_DIR,
  metricsPath = METRICS_PATH,
): Promise<string[]> {
  const charts = buildKdaCharts(loadMetrics(metricsPath))
  for (const chart of Object.values(charts)) parseChart(chart)
  mkdirSync(outputDir, { recursive: true })
  const paths: string[] = []
  for (const name of CHART_NAMES) {
    const path = resolve(outputDir, `${name}.json`)
    const options = (await resolveConfig(path)) ?? {}
    writeFileSync(path, await format(JSON.stringify(charts[name]), { ...options, filepath: path }))
    paths.push(path)
  }
  return paths
}

if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  for (const path of await writeKdaCharts()) console.log(path)
}
