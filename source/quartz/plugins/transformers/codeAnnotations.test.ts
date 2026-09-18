import assert from "node:assert/strict"
import { readFile } from "node:fs/promises"
import path from "node:path"
import { test } from "node:test"
import { fileURLToPath } from "node:url"
import { build } from "esbuild"
import { Element, Nodes, Root as HtmlRoot } from "hast"
import { toHtml } from "hast-util-to-html"
import { toString } from "hast-util-to-string"
import { Root } from "mdast"
import remarkParse from "remark-parse"
import remarkRehype from "remark-rehype"
import rehypeRaw from "rehype-raw"
import { unified } from "unified"
import { visit } from "unist-util-visit"
import { VFile } from "vfile"
import { BuildCtx } from "../../util/ctx"
import { Latex } from "./latex"
import { SyntaxHighlighting } from "./syntax"

// Match Quartz's inline-script loader, including bundling the real browser entrypoint.
const bundle = await build({
  entryPoints: [fileURLToPath(new URL("./codeAnnotations.ts", import.meta.url))],
  bundle: true,
  write: false,
  platform: "node",
  format: "esm",
  plugins: [
    {
      name: "quartz-inline-script",
      setup(builder) {
        builder.onLoad({ filter: /\.inline\.ts$/ }, async ({ path: entry }) => {
          const text = (await readFile(entry, "utf8"))
            .replace("export default", "")
            .replace("export", "")
          const script = await build({
            stdin: {
              contents: text,
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
const { CodeAnnotations } = (await import(
  `data:text/javascript;base64,${Buffer.from(bundle.outputFiles[0].text).toString("base64")}`
)) as typeof import("./codeAnnotations")

const ctx = {} as BuildCtx
const fence = (code: string, meta = "annotate", lang = "python") =>
  `\`\`\`${lang} ${meta}\n${code}\n\`\`\``
const input = (code = "x = 1 # (1)!", notes = "1. One", meta = "annotate", lang = "python") =>
  `${fence(code, meta, lang)}\n\n${notes}`

async function markdown(source: string) {
  const file = new VFile({ path: "/content/nested/annotations.md", value: source })
  const processor = unified().use(remarkParse).use(CodeAnnotations().markdownPlugins!(ctx))
  return (await processor.run(processor.parse(file), file)) as Root
}

async function render(source: string, annotations = true) {
  const file = new VFile({ path: "/content/nested/annotations.md", value: source })
  const processor = unified()
    .use(remarkParse)
    .use(Latex().markdownPlugins!(ctx))
    .use(annotations ? CodeAnnotations().markdownPlugins!(ctx) : [])
    .use(remarkRehype, { allowDangerousHtml: true })
    .use(SyntaxHighlighting().htmlPlugins!(ctx))
    .use(annotations ? CodeAnnotations().htmlPlugins!(ctx) : [])
    .use(rehypeRaw)
    .use(Latex().htmlPlugins!(ctx))
  const tree = (await processor.run(processor.parse(file), file)) as HtmlRoot
  return { tree, html: toHtml(tree, { allowDangerousHtml: true }) }
}

function elements(tree: HtmlRoot | Element, predicate: (element: Element) => boolean) {
  const matches: Element[] = []
  visit(tree, "element", (element) => {
    if (predicate(element)) matches.push(element)
  })
  return matches
}
const byClass = (tree: HtmlRoot | Element, name: string) =>
  elements(
    tree,
    (element) => (element.properties.className as string[] | undefined)?.includes(name) ?? false,
  )

const lines = (tree: HtmlRoot | Element) =>
  elements(tree, (element) => element.tagName === "span" && "dataLine" in element.properties)

function withoutPositions(tree: Nodes) {
  return JSON.parse(JSON.stringify(tree, (key, value) => (key === "position" ? undefined : value)))
}

test("all five marker comment forms produce clean source and accessible SSR anchors", async (t) => {
  for (const [lang, code, clean] of [
    ["python", "x = 1 # (1)!", "x = 1"],
    ["javascript", "const x = 1; // (1)!", "const x = 1;"],
    ["sql", "SELECT 1; -- (1)!", "SELECT 1;"],
    ["c", "int x = 1; /* (1)! */", "int x = 1;"],
    ["html", '<div class="example"></div> <!-- (1)! -->', '<div class="example"></div>'],
  ]) {
    await t.test(lang, async () => {
      const { tree, html } = await render(input(code, "1. One", "annotate", lang))
      const wrappers = byClass(tree, "code-annotations")
      assert.equal(wrappers.length, 1)
      assert.equal(wrappers[0].tagName, "div")
      assert.equal(wrappers[0].properties.dataCodeSource, clean)
      assert(!html.includes("(1)!"))
      assert(!html.includes("data-code-annotation-map"))
      const [marker] = byClass(tree, "code-annotation-marker")
      assert.equal(marker.tagName, "a")
      assert.deepEqual(marker.properties, {
        className: ["code-annotation-marker"],
        id: "code-annotation-marker-1-1",
        dataCodeNote: "1",
        href: "#code-note-1-1",
        dataRouterIgnore: "",
        dataNoPopover: "true",
        ariaLabel: "Read code annotation 1",
      })
      assert.equal(toString(marker), "1")
      assert.equal(lines(tree)[0].properties.dataCodeAnnotated, "")
      const [notes] = byClass(tree, "code-annotation-notes")
      assert.equal(notes.tagName, "ol")
      const [note] = notes.children.filter((child) => child.type === "element")
      assert.equal(note.tagName, "li")
      assert.equal(note.properties.id, "code-note-1-1")
      assert.equal(note.properties.dataCodeNote, "1")
      assert.equal(toString(note), "One")
      assert(!("hidden" in note.properties))

      const multiple = await render(
        input(code.replace("(1)!", "(1)! (2)!"), "1. One\n2. Two", "annotate", lang),
      )
      assert.deepEqual(
        byClass(multiple.tree, "code-annotation-marker").map(
          (element) => element.properties.dataCodeNote,
        ),
        ["1", "2"],
      )
      assert.equal(byClass(multiple.tree, "code-annotations")[0].properties.dataCodeSource, clean)
    })
  }
  const resources = CodeAnnotations().externalResources!(ctx)
  assert.equal(resources.js?.length, 1)
  const resource = resources.js![0]
  assert.equal(resource.contentType, "inline")
  assert.equal(resource.loadTime, "afterDOMReady")
  assert("script" in resource && typeof resource.script === "string" && resource.script.length > 0)
})

test("multiple/repeated references use stable block-local notes and unique marker IDs", async () => {
  const first = input("x = 1 # (1)! (2)!\nx += 1 # (1)!\nprint(x)", "1. One\n1. Two")
  const source = `${first}\n\n${input("x = 2 # (1)!")}`
  const { tree, html } = await render(source)
  assert.equal(byClass(tree, "code-annotations").length, 2)
  const markers = byClass(tree, "code-annotation-marker")
  assert.deepEqual(
    markers.map((marker) => marker.properties.href),
    ["#code-note-1-1", "#code-note-1-2", "#code-note-1-1", "#code-note-2-1"],
  )
  const ids = elements(tree, (element) => typeof element.properties.id === "string").map(
    (element) => element.properties.id,
  )
  assert.equal(new Set(ids).size, ids.length)
  assert.equal("dataCodeAnnotated" in lines(tree)[2].properties, false)
  assert.equal((await render(source)).html, html)
  assert((await render(input())).html.includes('id="code-note-1-1"'))
})

test("blank lines, comment-only lines and nested annotated fences retain their line/block maps", async () => {
  for (const code of ["# (1)!", "\n# (1)!\n", "x = 1\n  # (1)!\n\n"]) {
    const { tree } = await render(input(code))
    assert.equal(lines(tree).length, code.split("\n").length)
    assert.equal(byClass(tree, "code-annotation-marker").length, 1)
    assert.equal(
      byClass(tree, "code-annotations")[0].properties.dataCodeSource,
      code.replace(/ *# \(1\)!/, ""),
    )
  }
  const nested = input()
    .split("\n")
    .map((line) => `   ${line}`)
    .join("\n")
  const { tree } = await render(input(undefined, `1. Outer note\n\n${nested}`))
  assert.equal(byClass(tree, "code-annotations").length, 2)
  assert.deepEqual(
    byClass(tree, "code-annotation-marker").map((marker) => marker.properties.href),
    ["#code-note-1-1", "#code-note-2-1"],
  )
})

test("notes retain Markdown structure, links, math, nested lists and code blocks", async () => {
  const notes = [
    "1. **Bold** with *emphasis*, [a link](https://example.org/path), and `inline_code`.",
    "",
    "   A second paragraph with $x^2$.",
    "",
    "   - A nested item",
    "",
    "   ```python",
    "   print('note example') # (99)!",
    "   ```",
  ].join("\n")
  const { tree, html } = await render(input(undefined, notes))
  const [list] = byClass(tree, "code-annotation-notes")
  assert.equal(elements(list, (element) => element.tagName === "strong").length, 1)
  assert.equal(elements(list, (element) => element.tagName === "em").length, 1)
  assert.equal(elements(list, (element) => element.tagName === "ul").length, 1)
  assert(elements(list, (element) => element.tagName === "p").length >= 2)
  assert.equal(
    elements(list, (element) => element.properties.href === "https://example.org/path").length,
    1,
  )
  assert.equal(byClass(list, "katex").length, 1)
  assert(toString(list).includes("print('note example') # (99)!"))
  assert.equal(byClass(tree, "code-annotation-marker").length, 1)
  assert(html.includes('data-code-note="1"'))
})

test("only annotate metadata is removed; Shiki colors, titles, highlights and line numbers survive", async () => {
  const meta = 'title="annotate example.py" {2} showLineNumbers{10} /annotate/ annotate'
  const code = "x = 1 # (1)!\nprint(x) # (2)!"
  const notes = "1. One\n2. Two"
  const md = await markdown(input(code, notes, meta))
  const wrapper = md.children[0]
  assert(wrapper.type === "blockquote")
  const cleanCode = wrapper.children[0]
  assert(cleanCode.type === "code")
  assert.equal(cleanCode.meta, meta.slice(0, -"annotate".length))

  const annotated = await render(input(code, notes, meta))
  const ordinary = await render(fence("x = 1\nprint(x)", meta.slice(0, -"annotate".length)))
  const annotatedLines = lines(annotated.tree)
  for (const [index, line] of annotatedLines.entries()) {
    const original = lines(ordinary.tree)[index]
    assert.deepEqual(
      line.children.slice(0, -1).map(withoutPositions),
      original.children.map(withoutPositions),
    )
    const { dataCodeAnnotated: _annotated, ...properties } = line.properties
    assert.deepEqual(properties, original.properties)
  }
  assert(annotated.html.includes("--shiki-light:"))
  assert(annotated.html.includes("--shiki-dark:"))
  assert(annotated.html.includes("data-highlighted-line"))
  assert(annotated.html.includes("data-line-numbers"))
  assert(annotated.html.includes("counter-set: line 9"))
  assert(annotated.html.includes("annotate example.py"))
})

test("ordinary fences, lists, marker-looking text and quoted metadata remain unchanged", async () => {
  for (const meta of [
    "",
    'title="example annotate code"',
    "annotation",
    "/x annotate y/",
    '"annotate"',
  ]) {
    const source = input("x = 1 # (1)!", "1. Ordinary list", meta)
    assert.deepEqual(await markdown(source), unified().use(remarkParse).parse(source))
    assert.equal((await render(source)).html, (await render(source, false)).html)
  }
  const code = [
    'text = "# (1)!"',
    'url = "https://example.org/(1)!"',
    "x = 1 # prose about (99)!",
    "# ordinary comment  ",
    "x += 1 # (1)!",
    "",
  ].join("\n")
  const { tree } = await render(input(code))
  assert.equal(
    byClass(tree, "code-annotations")[0].properties.dataCodeSource,
    code.replace("x += 1 # (1)!", "x += 1"),
  )
  assert.equal(byClass(tree, "code-annotation-marker").length, 1)
})

test("reserved embed fences are untouched while unknown code languages get plaintext highlighting", async () => {
  for (const lang of ["chart", "plotly", "perfetto", "html-widget", "mermaid", "math"]) {
    const source = input("not an annotation", "1. Ordinary list", "annotate", lang)
    assert.deepEqual(await markdown(source), unified().use(remarkParse).parse(source))
    assert.equal((await render(source)).html, (await render(source, false)).html)
  }
  const { tree } = await render(
    input("custom = value # (1)!", "1. DSL note", "annotate", "future-dsl"),
  )
  assert.equal(byClass(tree, "code-annotation-marker").length, 1)
  assert.equal(byClass(tree, "code-annotations")[0].properties.dataCodeSource, "custom = value")
})

test("malformed opted-in blocks fail with the page filepath", async (t) => {
  const cases: [string, string][] = [
    [input("x = 1"), "marker-only comments"],
    [fence("x = 1 # (1)!"), "followed immediately by an ordered list"],
    [input(undefined, "- Unordered"), "ordered list"],
    [input(undefined, "Paragraph first\n\n1. One"), "ordered list"],
    [input(undefined, "2. Two"), "start at 1"],
    [input(undefined, "0. Zero"), "start at 1"],
    [input("x = 1 # (2)!"), "missing note 2"],
    [input(undefined, "1. One\n2. Unused"), "unused note 2"],
    [input("x = 1 # (2)!", "1. Unused\n2. Two"), "unused note 1"],
    ...["0", "-1", "1.5", "01", "wat", "9007199254740992"].map((number): [string, string] => [
      input(`x = 1 # (${number})!`),
      "invalid annotation number",
    ]),
  ]
  for (const [source, message] of cases) {
    await t.test(message, async () => {
      await assert.rejects(render(source), (error: Error) => {
        assert(error.message.startsWith("/content/nested/annotations.md: code annotations:"))
        assert(error.message.includes(message), error.message)
        return true
      })
    })
  }
})
