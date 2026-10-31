import { Properties } from "hast"
import { BlockContent, Code, Root, RootContent } from "mdast"
import { visit } from "unist-util-visit"
import { QuartzTransformerPlugin } from "../types"
// @ts-ignore
import script from "../../components/scripts/notebookCells.inline.ts"
import { takeFenceFlag } from "./fenceMeta"

// Render ```python cell fences as Jupyter-style input cells. An immediately following
// ```output fence becomes that cell's output; write ```output python to highlight it.
// The prose after a cell is its commentary and sits beside the output: it runs until the
// next cell or a heading that closes the section containing the cell.
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

const isCell = (node: RootContent | undefined) =>
  node?.type === "code" && takeFenceFlag(node.meta, "cell").present

function sectionDepth(siblings: RootContent[], position: number) {
  for (let index = position - 1; index >= 0; index--) {
    const node = siblings[index]
    if (node.type === "heading") return node.depth
  }
  // Outside any section, every heading ends the commentary.
  return 6
}

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

          const siblings = parent!.children as RootContent[]
          const children = [
            block("div", { className: ["nb-input"] }, [
              prompt("nb-prompt-in", `In [${count}]:`),
              node,
            ]),
          ]
          let end = position! + 1
          const next = siblings[end]
          if (next?.type === "code" && next.lang === outputLanguage) {
            const output = next as Code
            const [language, ...outputMeta] = (output.meta ?? "").trim().split(/\s+/)
            output.lang = language || "text"
            output.meta = outputMeta.join(" ") || null
            end++

            const depth = sectionDepth(siblings, position!)
            const commentaryStart = end
            while (
              end < siblings.length &&
              !isCell(siblings[end]) &&
              !(
                siblings[end].type === "heading" &&
                (siblings[end] as { depth: number }).depth <= depth
              )
            ) {
              end++
            }
            const commentary = siblings.slice(commentaryStart, end) as BlockContent[]
            const outputPane = block("div", { className: ["nb-output"] }, [
              prompt("nb-prompt-out", `Out[${count}]:`),
              output,
            ])
            children.push(
              commentary.length > 0
                ? block("div", { className: ["nb-explain"] }, [
                    block("div", { className: ["nb-commentary"] }, commentary),
                    outputPane,
                  ])
                : outputPane,
            )
          }
          siblings.splice(
            position!,
            end - position!,
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
