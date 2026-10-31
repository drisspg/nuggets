import { Properties } from "hast"
import { BlockContent, Code, Root } from "mdast"
import { visit } from "unist-util-visit"
import { QuartzTransformerPlugin } from "../types"
// @ts-ignore
import script from "../../components/scripts/notebookCells.inline.ts"
import { takeFenceFlag } from "./fenceMeta"

// Render ```python cell fences as Jupyter-style input cells. An immediately following
// ```output fence becomes that cell's output, shown beside the code on wide screens.
// Write ```output python to highlight the output as that language; it defaults to text.
const outputLanguage = "output"

const block = (hName: string, hProperties: Properties, children: BlockContent[]): BlockContent => ({
  type: "blockquote",
  data: { hName, hProperties },
  children,
})

const prompt = (className: string, value: string): BlockContent => ({
  type: "paragraph",
  data: { hName: "span", hProperties: { className: ["nb-prompt", className] } },
  children: [{ type: "text", value }],
})

export const NotebookCells: QuartzTransformerPlugin = () => ({
  name: "NotebookCells",
  markdownPlugins() {
    return [
      () => (tree: Root, file) => {
        const fail = (message: string): never => {
          throw new Error(`${file.path}: notebook cells: ${message}`)
        }
        let count = 0
        visit(tree, "code", (node, position, parent) => {
          if (node.lang === outputLanguage) {
            return fail("an output fence must immediately follow a cell fence")
          }
          const { present, meta } = takeFenceFlag(node.meta, "cell")
          if (!present) return
          node.meta = meta
          count++

          const children = [
            block("div", { className: ["nb-input"] }, [
              prompt("nb-prompt-in", `In [${count}]:`),
              node,
            ]),
          ]
          const next = parent!.children[position! + 1]
          const output =
            next?.type === "code" && next.lang === outputLanguage ? (next as Code) : null
          if (output) {
            const [language, ...meta] = (output.meta ?? "").trim().split(/\s+/)
            output.lang = language || "text"
            output.meta = meta.join(" ") || null
            children.push(
              block("div", { className: ["nb-output"] }, [
                prompt("nb-prompt-out", `Out[${count}]:`),
                output,
              ]),
            )
          }
          parent!.children.splice(
            position!,
            output ? 2 : 1,
            block("div", { className: ["nb-cell"], dataNbCount: String(count) }, children),
          )
          return position! + 1
        })
      },
    ]
  },
  externalResources() {
    // Transformer resources are classic scripts, unlike the isolated component bundles.
    return {
      js: [
        {
          script: `(function () {${script}})();`,
          loadTime: "afterDOMReady",
          contentType: "inline",
        },
      ],
    }
  },
})
