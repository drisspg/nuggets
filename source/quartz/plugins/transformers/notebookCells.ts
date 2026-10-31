import { Properties } from "hast"
import { BlockContent, Code, Emphasis, Root } from "mdast"
import { visit } from "unist-util-visit"
import { QuartzTransformerPlugin } from "../types"
import { takeFenceFlag } from "./fenceMeta"

// Render ```python cell fences as Jupyter-style input cells. An immediately following
// ```output fence becomes that cell's output; add `collapsed` to start it folded.
const outputLanguage = "output"

const label = (className: string[], value: string): Emphasis => ({
  type: "emphasis",
  data: { hName: "span", hProperties: { className } },
  children: [{ type: "text", value }],
})

const block = (hName: string, hProperties: Properties, children: BlockContent[]): BlockContent => ({
  type: "blockquote",
  data: { hName, hProperties },
  children,
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

          const children: BlockContent[] = [
            block("div", { className: ["nb-input"] }, [
              {
                type: "paragraph",
                data: { hName: "span", hProperties: { className: ["nb-prompt", "nb-prompt-in"] } },
                children: [{ type: "text", value: `In [${count}]:` }],
              },
              node,
            ]),
          ]
          const next = parent!.children[position! + 1]
          const output =
            next?.type === "code" && next.lang === outputLanguage ? (next as Code) : null
          if (output) {
            const { present: collapsed, meta: outputMeta } = takeFenceFlag(output.meta, "collapsed")
            const lines = output.value.split("\n").length
            output.lang = "text"
            output.meta = outputMeta
            children.push(
              block("details", { className: ["nb-output"], open: !collapsed }, [
                {
                  type: "paragraph",
                  data: { hName: "summary" },
                  children: [
                    label(["nb-prompt", "nb-prompt-out"], `Out[${count}]:`),
                    label(["nb-output-size"], `${lines} line${lines === 1 ? "" : "s"}`),
                  ],
                },
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
})
