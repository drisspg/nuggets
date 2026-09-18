import { Blockquote, Root } from "mdast"
import { Element, Root as HtmlRoot } from "hast"
import { visit } from "unist-util-visit"
import { QuartzTransformerPlugin } from "../types"
// @ts-ignore
import script from "../../components/scripts/codeAnnotations.inline.ts"

// Pair fences with Markdown notes before highlighting, then attach references to
// Shiki's line spans without replacing its syntax tokens or reparsing the notes.
const embedLanguages = new Set(["chart", "plotly", "perfetto", "html-widget", "mermaid", "math"])
// Quoted titles and slash-delimited character highlights may themselves contain "annotate".
const metaTokens = /(?:[^\s"'/]+|"(?:\\.|[^"\\])*"|'(?:\\.|[^'\\])*'|\/(?:\\.|[^/\\])*\/|\/)+/g
const markerComment =
  /(?:#|\/\/|--)[ \t]*((?:\([^()\r\n]*\)![ \t]*)+)$|\/\*[ \t]*((?:\([^()\r\n]*\)![ \t]*)+)\*\/[ \t]*$|<!--[ \t]*((?:\([^()\r\n]*\)![ \t]*)+)-->[ \t]*$/

interface AnnotationMap {
  block: number
  lines: number[][]
}

export const CodeAnnotations: QuartzTransformerPlugin = () => ({
  name: "CodeAnnotations",
  markdownPlugins() {
    return [
      () => (tree: Root, file) => {
        let block = 0
        visit(tree, "code", (node, position, parent) => {
          if (embedLanguages.has(node.lang ?? "")) return
          let optedIn = false
          const meta = node.meta?.replace(metaTokens, (token) => {
            if (token !== "annotate") return token
            optedIn = true
            return ""
          })
          if (!optedIn) return

          const fail = (message: string): never => {
            throw new Error(`${file.path}: code annotations: ${message}`)
          }
          const notes = parent!.children[position! + 1]
          if (notes?.type !== "list" || !notes.ordered) {
            return fail("an annotated fence must be followed immediately by an ordered list")
          }
          if (notes.start !== 1) fail("the annotation list must start at 1")

          const lines: number[][] = []
          const usedNotes = new Set<number>()
          const cleaned = node.value.split("\n").map((line) => {
            const comment = markerComment.exec(line)
            const numbers: number[] = []
            lines.push(numbers)
            if (!comment) return line
            const references = comment[1] ?? comment[2] ?? comment[3]
            for (const marker of references.matchAll(/\(([^()]*)\)!/g)) {
              const number = Number(marker[1])
              if (!/^[1-9]\d*$/.test(marker[1]) || !Number.isSafeInteger(number)) {
                fail(
                  `invalid annotation number ${JSON.stringify(marker[1])}; use positive integers`,
                )
              }
              if (number > notes.children.length) fail(`missing note ${number}`)
              numbers.push(number)
              usedNotes.add(number)
            }
            return line.slice(0, comment.index).trimEnd()
          })
          if (usedNotes.size === 0) fail("an annotated fence must contain marker-only comments")
          for (let number = 1; number <= notes.children.length; number++) {
            if (!usedNotes.has(number)) fail(`unused note ${number}`)
          }

          block++
          node.value = cleaned.join("\n")
          node.meta = meta
          notes.data = {
            ...notes.data,
            hProperties: { ...notes.data?.hProperties, className: ["code-annotation-notes"] },
          }
          for (const [index, item] of notes.children.entries()) {
            item.data = {
              ...item.data,
              hProperties: {
                ...item.data?.hProperties,
                id: `code-note-${block}-${index + 1}`,
                dataCodeNote: String(index + 1),
              },
            }
          }
          const wrapper: Blockquote = {
            type: "blockquote",
            data: {
              hName: "div",
              hProperties: {
                className: ["code-annotations"],
                dataCodeSource: node.value,
                dataCodeAnnotationMap: JSON.stringify({ block, lines } satisfies AnnotationMap),
              },
            },
            children: [node, notes],
          }
          parent!.children.splice(position!, 2, wrapper)
          return position
        })
      },
    ]
  },
  htmlPlugins() {
    return [
      () => (tree: HtmlRoot, file) => {
        visit(tree, "element", (wrapper) => {
          const mapping = wrapper.properties.dataCodeAnnotationMap
          if (typeof mapping !== "string") return
          const { block, lines } = JSON.parse(mapping) as AnnotationMap
          delete wrapper.properties.dataCodeAnnotationMap

          // Visit only this fence, not code examples (or nested annotations) inside its notes.
          const figure = wrapper.children.find(
            (child): child is Element =>
              child.type === "element" && ["figure", "pre"].includes(child.tagName),
          )
          const codeLines: Element[] = []
          if (figure) {
            visit(figure, "element", (element) => {
              if (
                element.tagName === "span" &&
                ("data-line" in element.properties || "dataLine" in element.properties)
              ) {
                codeLines.push(element)
              }
            })
          }
          if (codeLines.length !== lines.length) {
            throw new Error(
              `${file.path}: code annotations: expected highlighted code lines; place CodeAnnotations after SyntaxHighlighting`,
            )
          }
          let occurrence = 0
          for (const [index, numbers] of lines.entries()) {
            if (numbers.length === 0) continue
            const line = codeLines[index]
            line.properties.dataCodeAnnotated = ""
            for (const number of numbers) {
              line.children.push({
                type: "element",
                tagName: "a",
                properties: {
                  className: ["code-annotation-marker"],
                  id: `code-annotation-marker-${block}-${++occurrence}`,
                  dataCodeNote: String(number),
                  href: `#code-note-${block}-${number}`,
                  dataRouterIgnore: "",
                  dataNoPopover: "true",
                  ariaLabel: `Read code annotation ${number}`,
                },
                children: [{ type: "text", value: String(number) }],
              })
            }
          }
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
