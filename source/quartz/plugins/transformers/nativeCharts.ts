import { readFileSync, realpathSync } from "node:fs"
import path from "node:path"
import { Root } from "mdast"
import { visit } from "unist-util-visit"
import { QuartzTransformerPlugin } from "../types"
import { parseChart } from "../../util/chart"
import { escapeHTML } from "../../util/escape"
import { FilePath, pathToRoot, slugifyFilePath } from "../../util/path"
// @ts-ignore
import script from "../../components/scripts/nativeCharts.inline.ts"

export const NativeCharts: QuartzTransformerPlugin = () => ({
  name: "NativeCharts",
  markdownPlugins(ctx) {
    return [
      () => (tree: Root, file) => {
        visit(tree, "code", (node, position, parent) => {
          if (node.lang !== "chart") return
          let source: unknown
          try {
            const options: unknown = JSON.parse(node.value)
            if (!options || typeof options !== "object" || Array.isArray(options)) {
              throw new Error("options must be an object with src, title, and optional height")
            }
            const input = options as Record<string, unknown>
            source = input.src
            for (const key of Object.keys(input)) {
              if (!["src", "title", "height"].includes(key))
                throw new Error(`unknown chart option: ${key}`)
            }
            if (typeof input.title !== "string" || !input.title.trim()) {
              throw new Error("title must be a nonempty string")
            }
            const height = "height" in input ? input.height : 300
            if (
              typeof height !== "number" ||
              !Number.isInteger(height) ||
              height < 180 ||
              height > 800
            ) {
              throw new Error("height must be an integer from 180 to 800")
            }
            if (
              typeof source !== "string" ||
              !source.startsWith("media/") ||
              /[\\?#%\u0000-\u001f\u007f]/.test(source) ||
              source.split("/").some((part) => !part || part === "." || part === "..") ||
              path.extname(source) !== ".json"
            ) {
              throw new Error(
                "src must be a content-root-relative .json path inside media/ without traversal, query, or hash",
              )
            }
            const contentRoot = realpathSync(ctx.argv.directory)
            const dataPath = realpathSync(path.resolve(contentRoot, source))
            const relative = path.relative(contentRoot, dataPath)
            if (
              relative === ".." ||
              relative.startsWith(`..${path.sep}`) ||
              path.isAbsolute(relative)
            ) {
              throw new Error("src symlink escapes the content root")
            }
            parseChart(JSON.parse(readFileSync(dataPath, "utf8")))
            const emitted = slugifyFilePath(source as FilePath, true) + path.extname(source)
            const src = escapeHTML(
              `${pathToRoot(file.data.slug!)}/${emitted.split("/").map(encodeURIComponent).join("/")}`,
            )
            const title = escapeHTML(input.title)
            const html = `<figure class="native-chart" data-chart-src="${src}" data-chart-height="${height}" aria-label="${title}"><figcaption>${title}</figcaption><div class="native-chart-content"><p role="status">Loading chart…</p></div><noscript>Interactive chart requires JavaScript.</noscript><a class="native-chart-download" href="${src}" download data-router-ignore>Download chart data</a></figure>`
            parent!.children.splice(position!, 1, { type: "html", value: html })
          } catch (error) {
            const message = error instanceof Error ? error.message : String(error)
            throw new Error(
              `${file.path}: chart${source === undefined ? "" : ` ${JSON.stringify(source)}`}: ${message}`,
              { cause: error },
            )
          }
        })
      },
    ]
  },
  externalResources() {
    return { js: [{ script, loadTime: "afterDOMReady", contentType: "inline" }] }
  },
})
