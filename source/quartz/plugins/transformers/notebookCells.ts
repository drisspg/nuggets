import { Properties } from "hast"
import { BlockContent, Code, Heading, Root, RootContent } from "mdast"
import { visit } from "unist-util-visit"
import { QuartzTransformerPlugin } from "../types"
// @ts-ignore
import script from "../../components/scripts/notebookCells.inline.ts"
import { takeFenceFlag } from "./fenceMeta"

// Render ```python cell fences as Jupyter-style input cells. An immediately following
// ```output fence becomes that cell's output; write ```output python to highlight it.
// A cell with output gathers the prose around it: the lead-in since the last heading and
// the commentary after it. On wide screens that prose and the code form the left column
// beside a pinned log.
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

const cellBlock = (count: number, children: BlockContent[]) =>
  block("div", { className: ["nb-cell"], dataNbCount: String(count) }, children)

// Lead-in prose never reaches past a heading or an already-built cell.
const isBoundary = (node: RootContent) =>
  node.type === "heading" ||
  isCell(node) ||
  ((node.data?.hProperties?.className as string[] | undefined)?.includes("nb-cell") ?? false)

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
        // Notebook Markdown often leaves fences untagged; a page can name their language.
        const fenceLanguage = file.data.frontmatter?.notebookFenceLanguage
        if (typeof fenceLanguage === "string") {
          visit(tree, "code", (node) => {
            node.lang ??= fenceLanguage
          })
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
          const input = block("div", { className: ["nb-input"] }, [
            prompt("nb-prompt-in", `In [${count}]:`),
            node,
          ])
          let start = position!
          let end = position! + 1
          const next = siblings[end]
          if (!(next?.type === "code" && next.lang === outputLanguage)) {
            siblings.splice(start, 1, cellBlock(count, [input]))
            return start + 1
          }
          const output = next as Code
          const [language, ...outputMeta] = (output.meta ?? "").trim().split(/\s+/)
          output.lang = language || "text"
          output.meta = outputMeta.join(" ") || null
          end++

          // Lead-in: the prose since the last heading or cell. Commentary: the prose after the
          // output until the next cell or a heading that closes the cell's section. A subheading
          // that leads straight into the next cell introduces that cell, so leave it outside.
          while (start > 0 && !isBoundary(siblings[start - 1])) start--
          const depth = sectionDepth(siblings, start)
          const commentaryStart = end
          while (
            end < siblings.length &&
            !isCell(siblings[end]) &&
            !(siblings[end].type === "heading" && (siblings[end] as Heading).depth <= depth)
          ) {
            end++
          }
          if (isCell(siblings[end])) {
            const lastHeading = siblings
              .slice(commentaryStart, end)
              .findLastIndex((node) => node.type === "heading")
            if (lastHeading !== -1) end = commentaryStart + lastHeading
          }
          const prose = (from: number, to: number, className: string) =>
            to > from
              ? [
                  block(
                    "div",
                    { className: [className] },
                    siblings.slice(from, to) as BlockContent[],
                  ),
                ]
              : []
          siblings.splice(
            start,
            end - start,
            cellBlock(count, [
              ...prose(start, position!, "nb-lead"),
              input,
              block("div", { className: ["nb-output"] }, [
                prompt("nb-prompt-out", `Out[${count}]:`),
                output,
              ]),
              ...prose(commentaryStart, end, "nb-commentary"),
            ]),
          )
          // Revisit the new cell so nested fences in its prose are converted or rejected too.
          return start
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
