import assert from "node:assert/strict"
import { mkdtemp, mkdir, readFile, rm, symlink, writeFile } from "node:fs/promises"
import os from "node:os"
import path from "node:path"
import { test, TestContext } from "node:test"
import { fileURLToPath } from "node:url"
import { build } from "esbuild"
import { Root } from "mdast"
import remarkParse from "remark-parse"
import { unified } from "unified"
import { VFile } from "vfile"
import { BuildCtx } from "../../util/ctx"
import { FullSlug } from "../../util/path"

// Quartz imports browser entrypoints as bundled text, not as executable Node modules.
// Compile the real entrypoint here too, so a missing/broken renderer fails the suite.
const bundle = await build({
  entryPoints: [fileURLToPath(new URL("./nativeCharts.ts", import.meta.url))],
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
const { NativeCharts } = (await import(
  `data:text/javascript;base64,${Buffer.from(bundle.outputFiles[0].text).toString("base64")}`
)) as typeof import("./nativeCharts")

const data = {
  version: 1,
  x: { label: "Step" },
  y: { label: "Loss" },
  series: [{ name: "<script>untrusted data</script>", mode: "line", points: [{ x: 1, y: 2 }] }],
}
const options = { src: "media/chart.json", title: "Training loss" }
const fence = (value: unknown, lang = "chart") => `\`\`\`${lang}\n${JSON.stringify(value)}\n\`\`\``

async function fixture(t: TestContext) {
  const temporary = await mkdtemp(path.join(os.tmpdir(), "quartz-native-chart-"))
  t.after(() => rm(temporary, { recursive: true, force: true }))
  const root = path.join(temporary, "content")
  await mkdir(path.join(root, "media"), { recursive: true })
  await writeFile(path.join(root, options.src), JSON.stringify(data))
  return { root, temporary }
}

async function transform(directory: string, markdown: string, slug = "note") {
  const ctx = { argv: { directory } } as BuildCtx
  const file = new VFile({ path: path.join(directory, `${slug}.md`), value: markdown })
  file.data.slug = slug as FullSlug
  const processor = unified().use(remarkParse).use(NativeCharts().markdownPlugins!(ctx))
  return (await processor.run(processor.parse(file), file)) as Root
}

async function render(directory: string, input: unknown = options, slug = "note") {
  const tree = await transform(directory, fence(input), slug)
  assert.equal(tree.children.length, 1)
  const node = tree.children[0]
  assert(node.type === "html")
  return node.value
}

test("real Markdown transform emits the exact accessible shell and download contract", async (t) => {
  const { root } = await fixture(t)
  assert.equal(
    await render(root),
    '<figure class="native-chart" data-chart-src="./media/chart.json" data-chart-height="300" aria-label="Training loss"><figcaption>Training loss</figcaption><div class="native-chart-content"><p role="status">Loading chart…</p></div><noscript>Interactive chart requires JavaScript.</noscript><a class="native-chart-download" href="./media/chart.json" download data-router-ignore>Download chart data</a></figure>',
  )
  const resources = NativeCharts().externalResources!({} as BuildCtx)
  assert.equal(resources.js?.length, 1)
  const resource = resources.js![0]
  assert.equal(resource.contentType, "inline")
  assert.equal(resource.loadTime, "afterDOMReady")
  assert("script" in resource && typeof resource.script === "string" && resource.script.length > 0)
})

test("root/nested notes and deployed URL prefixes resolve to content-root media", async (t) => {
  const { root } = await fixture(t)
  for (const [slug, page, relative] of [
    ["index", "https://example.test/garden/", "./media/chart.json"],
    ["nested/note", "https://example.test/garden/nested/note", "../media/chart.json"],
    ["nested/deep/index", "https://example.test/garden/nested/deep/", "../../media/chart.json"],
  ]) {
    const html = await render(path.relative(process.cwd(), root), options, slug)
    assert(html.includes(`data-chart-src="${relative}"`))
    assert(html.includes(`href="${relative}"`))
    assert.equal(new URL(relative, page).href, "https://example.test/garden/media/chart.json")
  }
})

test("escapes title and emitted asset URL without embedding untrusted data", async (t) => {
  const { root } = await fixture(t)
  const src = `media/it's "quoted" <data>.json`
  await writeFile(path.join(root, src), JSON.stringify(data))
  const html = await render(root, { src, title: `"<img src=x onerror='bad()'> & Loss` })
  assert(html.includes('aria-label="&quot;&lt;img src=x onerror=&#039;bad()&#039;&gt; &amp; Loss"'))
  assert(
    html.includes(
      "<figcaption>&quot;&lt;img src=x onerror=&#039;bad()&#039;&gt; &amp; Loss</figcaption>",
    ),
  )
  assert(html.includes('data-chart-src="./media/it&#039;s-%22quoted%22-%3Cdata%3E.json"'))
  assert(!html.includes("<img"))
  assert(!html.includes("<script>"))
  assert(!html.includes("untrusted data"))
})

test("supports boundary heights and transforms adjacent chart fences", async (t) => {
  const { root } = await fixture(t)
  const markdown = [180, 800].map((height) => fence({ ...options, height })).join("\n\n")
  const tree = await transform(root, markdown)
  assert.equal(tree.children.length, 2)
  for (const [index, height] of [180, 800].entries()) {
    const node = tree.children[index]
    assert(node.type === "html")
    assert(node.value.includes(`data-chart-height="${height}"`))
  }
})

test("legacy and non-chart fences are untouched", async (t) => {
  const { root } = await fixture(t)
  const markdown = ["plotly", "perfetto", "html-widget", "json", "Chart", ""]
    .map((lang) => fence("not chart options", lang))
    .join("\n\n")
  assert.deepEqual(await transform(root, markdown), unified().use(remarkParse).parse(markdown))
})

test("rejects malformed JSON and unknown/malformed options with the note path", async (t) => {
  const { root } = await fixture(t)
  await assert.rejects(transform(root, "```chart\n{\n```"), (error: Error) =>
    error.message.includes(`${root}/note.md: chart:`),
  )
  for (const input of [
    null,
    [],
    "text",
    { ...options, renderer: "plotly" },
    { src: options.src },
    { ...options, title: " " },
    { ...options, title: 1 },
  ]) {
    await assert.rejects(render(root, input), /note\.md: chart/)
  }
  for (const height of [179, 801, 300.5, "300", null, false]) {
    await assert.rejects(
      render(root, { ...options, height }),
      /note\.md: chart "media\/chart\.json": height must be an integer from 180 to 800/,
    )
  }
})

test("rejects unsafe or non-JSON media paths before reading files", async (t) => {
  const { root } = await fixture(t)
  for (const src of [
    "https://example.test/chart.json",
    "/media/chart.json",
    "//example.test/chart.json",
    "chart.json",
    "media/../chart.json",
    "media/./chart.json",
    "media//chart.json",
    "media/chart.json?raw=1",
    "media/chart.json#fragment",
    "media/%2e%2e/chart.json",
    "media/chart%20data.json",
    "media\\chart.json",
    "media/chart.html",
    "media/chart.JSON",
    "media/\u0000chart.json",
    "media/chart.json/",
    null,
    1,
  ]) {
    await t.test(JSON.stringify(src), () =>
      assert.rejects(
        render(root, { ...options, src }),
        /note\.md: chart .*src must be a content-root-relative \.json path inside media/,
      ),
    )
  }
})

test("rejects escaping file/directory symlinks but accepts an in-root symlink", async (t) => {
  const { root, temporary } = await fixture(t)
  await writeFile(path.join(temporary, "outside.json"), JSON.stringify(data))
  await symlink(path.join(temporary, "outside.json"), path.join(root, "media/outside.json"))
  await symlink(temporary, path.join(root, "media/outside-directory"))
  for (const src of ["media/outside.json", "media/outside-directory/outside.json"]) {
    await assert.rejects(
      render(root, { ...options, src }),
      /note\.md: chart .*src symlink escapes the content root/,
    )
  }
  await symlink(path.join(root, options.src), path.join(root, "media/inside.json"))
  assert(
    (await render(root, { ...options, src: "media/inside.json" })).includes(
      'data-chart-src="./media/inside.json"',
    ),
  )
})

test("missing files, malformed data JSON, and schema failures identify note and data source", async (t) => {
  const { root } = await fixture(t)
  await assert.rejects(
    render(root, { ...options, src: "media/missing.json" }),
    /note\.md: chart "media\/missing\.json": ENOENT/,
  )
  const file = path.join(root, options.src)
  await writeFile(file, "not JSON")
  await assert.rejects(render(root), /note\.md: chart "media\/chart\.json":/)
  await writeFile(
    file,
    JSON.stringify({ ...data, series: [{ name: "A", mode: "line", points: [{ x: "1", y: 2 }] }] }),
  )
  await assert.rejects(
    render(root),
    /note\.md: chart "media\/chart\.json": chart\.series\[0\]\.points\[0\]\.x: expected a finite number/,
  )
  await writeFile(file, JSON.stringify(data))
  assert((await render(root)).includes('class="native-chart"'))
})
