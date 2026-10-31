import assert from "node:assert/strict"
import { readFile } from "node:fs/promises"
import path from "node:path"
import { test } from "node:test"
import { fileURLToPath } from "node:url"
import { runInNewContext } from "node:vm"
import { build } from "esbuild"
import { Element, Root as HtmlRoot } from "hast"
import { toString } from "hast-util-to-string"
import remarkParse from "remark-parse"
import remarkRehype from "remark-rehype"
import { unified } from "unified"
import { visit } from "unist-util-visit"
import { VFile } from "vfile"
import { BuildCtx } from "../../util/ctx"
import { SyntaxHighlighting } from "./syntax"

// Match Quartz's inline-script loader, including bundling the real browser entrypoint.
const bundle = await build({
  entryPoints: [fileURLToPath(new URL("./notebookCells.ts", import.meta.url))],
  bundle: true,
  write: false,
  platform: "node",
  format: "esm",
  plugins: [
    {
      name: "quartz-inline-script",
      setup(builder) {
        builder.onLoad({ filter: /\.inline\.ts$/ }, async ({ path: entry }) => {
          const script = await build({
            stdin: {
              contents: await readFile(entry, "utf8"),
              loader: "ts",
              resolveDir: path.dirname(entry),
              sourcefile: entry,
            },
            bundle: true,
            write: false,
            minify: true,
            platform: "browser",
            format: "esm",
          })
          return { contents: script.outputFiles[0].text, loader: "text" }
        })
      },
    },
  ],
})
const { NotebookCells } = (await import(
  `data:text/javascript;base64,${Buffer.from(bundle.outputFiles[0].text).toString("base64")}`
)) as typeof import("./notebookCells")

test("the bundled prose linker does not leak minified bindings into other classic scripts", () => {
  const resource = NotebookCells().externalResources!({} as BuildCtx).js![0]
  assert("script" in resource)
  const sandbox = { document: { addEventListener() {} } }
  runInNewContext(resource.script, sandbox)
  assert.deepEqual(Object.keys(sandbox), ["document"])
})

const ctx = {} as BuildCtx

async function render(source: string, frontmatter: Record<string, unknown> = {}) {
  const file = new VFile({ path: "/content/nested/notebook.md", value: source })
  file.data.frontmatter = frontmatter as typeof file.data.frontmatter
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

test("a cell with output gathers its lead-in and commentary until its section closes", async () => {
  const tree = await render(
    [
      "# Section",
      "",
      "Lead-in.",
      "",
      "```python cell",
      "run()",
      "```",
      "",
      "```output",
      "log",
      "```",
      "",
      "About the log.",
      "",
      "## Subsection",
      "",
      "Still about it.",
      "",
      "# Next section",
      "",
      "Unrelated.",
      "",
      "## Bare",
      "",
      "```python cell",
      "again()",
      "```",
      "",
      "```output",
      "log",
      "```",
      "",
      "## Sibling heading",
    ].join("\n"),
  )
  const [first, second] = byClass(tree, "nb-cell")
  assert.equal(toString(byClass(first, "nb-lead")[0]).trim(), "Lead-in.")
  // The heading that opens the section stays outside the cell.
  assert.doesNotMatch(toString(first), /^\s*Section/)
  const commentary = byClass(first, "nb-commentary")
  assert.equal(commentary.length, 1)
  assert.match(toString(commentary[0]), /About the log\.[\s\S]*Subsection[\s\S]*Still about it\./)
  assert.doesNotMatch(toString(first), /Next section|Unrelated/)
  // A heading at the cell's own level closes it immediately, and nothing leads in after one.
  assert.equal(byClass(second, "nb-commentary").length + byClass(second, "nb-lead").length, 0)
  assert.equal(byClass(second, "nb-output").length, 1)
  assert.match(toString(tree), /Sibling heading/)
  assert.doesNotMatch(toString(second), /Sibling heading/)
})

test("an output fence may name its highlighting language", async () => {
  const tree = await render("```python cell\nrun()\n```\n\n```output python\nx = 1\n```")
  const [output] = byClass(tree, "nb-output")
  const pre = byClass(output, "nb-prompt")[0]
  assert.equal(toString(pre), "Out[1]:")
  let language: unknown
  visit(output, "element", (element) => {
    if (element.tagName === "pre") language = element.properties["data-language"]
  })
  assert.equal(language, "python")
})

test("a page can name the language of its untagged fences", async () => {
  const tree = await render("```\nx = 1\n```\n\n```text\nplain\n```", {
    notebookFenceLanguage: "python",
  })
  const languages: unknown[] = []
  visit(tree, "element", (element) => {
    if (element.tagName === "pre") languages.push(element.properties["data-language"])
  })
  assert.deepEqual(languages, ["python", "text"])
})

test("an output fence without a cell fails with the page filepath", async () => {
  await assert.rejects(
    render("```python\nrun()\n```\n\n```output\nx\n```"),
    /\/content\/nested\/notebook\.md: notebook cells: an output fence must immediately follow a cell fence/,
  )
})
