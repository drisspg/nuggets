import assert from "node:assert/strict"
import { test } from "node:test"
import { Element, Root as HtmlRoot } from "hast"
import { toString } from "hast-util-to-string"
import remarkParse from "remark-parse"
import remarkRehype from "remark-rehype"
import { unified } from "unified"
import { visit } from "unist-util-visit"
import { VFile } from "vfile"
import { BuildCtx } from "../../util/ctx"
import { NotebookCells } from "./notebookCells"
import { SyntaxHighlighting } from "./syntax"

const ctx = {} as BuildCtx

async function render(source: string) {
  const file = new VFile({ path: "/content/nested/notebook.md", value: source })
  const processor = unified()
    .use(remarkParse)
    .use(NotebookCells().markdownPlugins!(ctx))
    .use(remarkRehype)
    .use(SyntaxHighlighting().htmlPlugins!(ctx))
  return (await processor.run(processor.parse(file), file)) as HtmlRoot
}

const byClass = (tree: HtmlRoot | Element, name: string) => {
  const matches: Element[] = []
  visit(tree, "element", (element) => {
    if ((element.properties.className as string[] | undefined)?.includes(name)) {
      matches.push(element)
    }
  })
  return matches
}

test("cells are numbered in page order and outputs attach to the preceding cell", async () => {
  const tree = await render(
    [
      '```python cell title="first.py"',
      "print(1)",
      "```",
      "",
      "```output",
      "1",
      "```",
      "",
      "Some prose.",
      "",
      "```python cell",
      "x = 2",
      "```",
      "",
      "```python",
      "plain = True",
      "```",
    ].join("\n"),
  )
  const cells = byClass(tree, "nb-cell")
  assert.deepEqual(
    cells.map((cell) => cell.properties.dataNbCount),
    ["1", "2"],
  )
  assert.deepEqual(
    byClass(tree, "nb-prompt").map((prompt) => toString(prompt)),
    ["In [1]:", "Out[1]:", "In [2]:"],
  )
  const [output] = byClass(tree, "nb-output")
  assert.equal(output.tagName, "details")
  assert.equal(output.properties.open, true)
  assert.match(toString(output), /\b1\b/)
  assert.equal(byClass(cells[1], "nb-output").length, 0)
  // The cell flag is consumed while other fence metadata keeps working.
  assert.equal(toString(byClass(cells[0], "nb-input")[0]).includes("first.py"), true)
  assert.equal(toString(tree).includes("plain = True"), true)
  assert.equal(
    cells.some((cell) => toString(cell).includes("plain = True")),
    false,
  )
})

test("collapsed outputs start closed and report their size", async () => {
  const tree = await render("```python cell\nrun()\n```\n\n```output collapsed\na\nb\nc\n```")
  const [output] = byClass(tree, "nb-output")
  assert.notEqual(output.properties.open, true)
  assert.equal(toString(byClass(output, "nb-output-size")[0]), "3 lines")
})

test("an output fence without a cell fails with the page filepath", async () => {
  await assert.rejects(
    render("```python\nrun()\n```\n\n```output\nx\n```"),
    /\/content\/nested\/notebook\.md: notebook cells: an output fence must immediately follow a cell fence/,
  )
})
