import { writeFile } from "node:fs/promises"
import { resolve } from "node:path"
import { fileURLToPath } from "node:url"
import { format } from "prettier"
import type { ChartSpec } from "../../quartz/util/chart"

export const GATE_MAGNITUDE = 5 * Math.LOG2E
export const RANGE_PATH = fileURLToPath(
  new URL("../../content/media/kda/rebase-range.json", import.meta.url),
)

export function buildRebaseRange(): ChartSpec {
  const windows = Array.from({ length: 64 }, (_, i) => i + 1)
  return {
    version: 1,
    x: { label: "Window size (tokens)", domain: [1, 64] },
    y: { label: "Gate-factor exponent (base 2)", domain: [-480, 480] },
    series: ([1, -1] as const).map((sign) => ({
      name: sign === 1 ? "Inverse factor" : "Decay factor",
      mode: "line" as const,
      color: sign === 1 ? ("amber" as const) : ("green" as const),
      points: windows.map((window) => {
        const distance = window - 1
        const exponent = distance === 0 ? 0 : sign * distance * GATE_MAGNITUDE
        const factor = 2 ** exponent
        const fp32 = Math.fround(factor)
        return {
          x: window,
          y: exponent,
          details: {
            "Steps from first row": distance,
            "Unrounded factor": factor,
            "FP32 cast": Number.isFinite(fp32) ? fp32 : "Overflow",
          },
        }
      }),
    })),
    // FP32 cutoffs are reference barriers, not measured or analytic series.
    references: [
      { y: 128, label: "FP32 overflow (+128)", color: "red" },
      { y: -126, label: "FP32 subnormal (−126)", color: "purple" },
    ],
  }
}

if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  await writeFile(RANGE_PATH, await format(JSON.stringify(buildRebaseRange()), { parser: "json" }))
  console.log(`Wrote ${RANGE_PATH}`)
}
