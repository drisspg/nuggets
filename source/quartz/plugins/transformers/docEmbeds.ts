import { Root } from "mdast"
import { visit } from "unist-util-visit"
import { QuartzTransformerPlugin } from "../types"
import { escapeHTML } from "../../util/escape"
import path from "node:path"
import { FilePath, pathToRoot, slugifyFilePath } from "../../util/path"
// @ts-ignore
import script from "../../components/scripts/docEmbeds.inline.ts"

/** Render attention-gym-style helpers without introducing a Jinja runtime. */
export const DocEmbeds: QuartzTransformerPlugin = () => ({
  name: "DocEmbeds",
  markdownPlugins() {
    return [
      () => (tree: Root, file) => {
        let index = 0
        visit(tree, "code", (node, position, parent) => {
          if (!["perfetto", "plotly", "html-widget"].includes(node.lang ?? "")) return
          const options = JSON.parse(node.value)
          if (!options || typeof options.title !== "string" || !options.title.trim()) {
            throw new Error(`${file.path}: ${node.lang} requires a title`)
          }
          const root = pathToRoot(file.data.slug!)
          /** Asset paths are content-root-relative, never remote URLs or traversal. */
          const asset = (value: unknown) => {
            if (
              typeof value !== "string" ||
              !value.startsWith("media/") ||
              /[\\?#%]/.test(value) ||
              value.split("/").some((part) => !part || part === "." || part === "..")
            ) {
              throw new Error(`${file.path}: embed assets must be paths inside media/`)
            }
            const emitted = slugifyFilePath(value as FilePath, true) + path.extname(value)
            return `${root}/${emitted.split("/").map(encodeURIComponent).join("/")}`
          }
          const src = asset(options.src)
          const title = escapeHTML(options.title)
          const height = options.height ?? (node.lang === "perfetto" ? 680 : 560)
          if (!Number.isInteger(height) || height < 1)
            throw new Error(`${file.path}: embed height must be a positive integer`)
          const frame = `class="doc-widget widget-frame${node.lang === "plotly" ? " plotly-chart__frame" : ""}" title="${title}" loading="lazy" style="height:${height}px"`
          let html: string
          if (node.lang === "perfetto") {
            const snapshot = asset(options.snapshot)
            if (typeof options.alt !== "string" || !options.alt.trim())
              throw new Error(`${file.path}: perfetto requires snapshot alt text`)
            const id = `trace-viewer-${index++}`
            // Resolve the trace relative to the static widget, not the article's depth.
            const trace = `../../../${src.slice(root.length + 1)}`
            const viewer = `${root}/static/widgets/perfetto-trace/index.html?${new URLSearchParams({ trace, title: options.title })}`
            html = `<figure class="trace-embed" data-trace-embed><button class="trace-preview" type="button" aria-expanded="false" aria-controls="${id}"><img class="widget-frame" src="${snapshot}" alt="${escapeHTML(options.alt)}" loading="lazy"><span>Click to interact</span></button><div id="${id}" class="trace-embed__viewer" hidden><div class="trace-embed__controls"><button type="button" data-trace-close>Show snapshot</button><button type="button" data-trace-fullscreen aria-pressed="false">Fullscreen</button></div><iframe ${frame} data-src="${escapeHTML(viewer)}"></iframe></div><figcaption>${title} · <a href="${src}" download data-router-ignore>Download trace</a></figcaption></figure>`
          } else {
            html = `<figure class="${node.lang === "plotly" ? "plotly-chart" : "html-widget"}"><iframe ${frame} src="${src}" data-router-ignore></iframe><figcaption>${title}</figcaption></figure>`
          }
          parent!.children.splice(position!, 1, { type: "html", value: html })
        })
      },
    ]
  },
  externalResources() {
    return { js: [{ script, loadTime: "afterDOMReady", contentType: "inline" }] }
  },
})
