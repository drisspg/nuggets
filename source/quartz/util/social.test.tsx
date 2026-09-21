import assert from "node:assert/strict"
import fs from "node:fs/promises"
import os from "node:os"
import path from "node:path"
import test from "node:test"
import { render } from "preact-render-to-string"
import { unified } from "unified"
import remarkParse from "remark-parse"
import { VFile } from "vfile"
import { FrontMatter } from "../plugins/transformers/frontmatter"
import { SocialImage } from "../plugins/transformers/socialImage"
import Head from "../components/Head"
import { QuartzComponentProps } from "../components/types"
import { GlobalConfiguration } from "../cfg"
import { defaultProcessedContent } from "../plugins/vfile"
import { writeSocialImage } from "../plugins/emitters/socialImages"
import { BuildCtx } from "./ctx"
import { FullSlug } from "./path"
import { SOCIAL_HEIGHT, SOCIAL_WIDTH, socialMetadata } from "./social"

const cfg = {
  pageTitle: "Nuggets",
  baseUrl: "drisspg.github.io/nuggets",
  locale: "en-US",
  theme: { fontOrigin: "local", cdnCaching: false },
} as GlobalConfiguration

function page(slug = "KDA-Future-Token-Leakage", title = "KDA Doesn't Care About the Future") {
  return defaultProcessedContent({
    slug: slug as FullSlug,
    frontmatter: {
      title,
      tags: ["pytorch"],
      description: "Can a model learn from a tiny peek into future tokens?",
    },
    description: "Written: September 15, 2026 Product pitch",
  })[1].data
}

test("social copy uses explicit descriptions, then dek, then the automatic excerpt", () => {
  const file = page()
  assert.equal(socialMetadata(cfg, file).description, file.frontmatter!.description)
  delete file.frontmatter!.description
  file.frontmatter!.dek = "A deliberate subtitle."
  assert.equal(socialMetadata(cfg, file).description, "A deliberate subtitle.")
  delete file.frontmatter!.dek
  assert.equal(socialMetadata(cfg, file).description, file.description)
  delete file.description
  assert.match(socialMetadata(cfg, file).description, /Notes on PyTorch/)
})

test("canonical and image URLs retain the deployment prefix and escape nested slugs", () => {
  const card = socialMetadata(cfg, page("nested/Café-&-kernels"))
  assert.equal(card.url, "https://drisspg.github.io/nuggets/nested/Caf%C3%A9-&-kernels")
  assert.equal(card.imageUrl, `https://drisspg.github.io/nuggets/${card.imageSlug}.png`)
  assert.equal(socialMetadata(cfg, page("index")).url, "https://drisspg.github.io/nuggets/")
  assert.equal(
    socialMetadata(cfg, page("notes/index")).url,
    "https://drisspg.github.io/nuggets/notes/",
  )
  assert.equal(
    socialMetadata({ ...cfg, baseUrl: "example.com/" }, page()).url,
    "https://example.com/KDA-Future-Token-Leakage",
  )
  assert.equal(socialMetadata({ ...cfg, baseUrl: undefined }, page()).imageUrl, undefined)
})

test("cards are stable, distinct, and invalidate the image URL when social copy changes", () => {
  const file = page()
  const before = socialMetadata(cfg, file)
  assert.deepEqual(socialMetadata(cfg, file), before)
  file.frontmatter!.description = "Updated copy."
  assert.notEqual(socialMetadata(cfg, file).imageSlug, before.imageSlug)
  assert.equal(before.artwork, "attention")
  assert.equal(socialMetadata(cfg, page("index", "Welcome")).title, "Nuggets")
  assert.equal(socialMetadata(cfg, page("index")).type, "website")
  assert.equal(socialMetadata(cfg, page("tags/pytorch", "PyTorch")).type, "website")
  assert.equal(before.type, "article")
})

test("Head emits an explicit large-image card, canonical URL, dimensions and alt text in static HTML", () => {
  const file = page()
  file.frontmatter!.title = 'Q & K: <precision> "matters"'
  const component = Head()
  const props = {
    cfg,
    fileData: file,
    externalResources: { css: [], js: [] },
  } as unknown as QuartzComponentProps
  const html = render(<>{component(props)}</>)
  const card = socialMetadata(cfg, file)
  assert.match(html, /name="twitter:card" content="summary_large_image"/)
  assert.match(html, /name="twitter:creator" content="@drisspg"/)
  assert.match(html, /property="og:image:width" content="1200"/)
  assert.match(html, /property="og:image:height" content="630"/)
  assert.match(html, /property="og:image:type" content="image\/png"/)
  assert.match(html, /property="og:image:alt"/)
  assert.match(html, /name="twitter:image:alt"/)
  assert(html.includes(`property="og:image" content="${card.imageUrl}"`))
  assert(html.includes(`name="twitter:image" content="${card.imageUrl}"`))
  assert(html.includes(`rel="canonical" href="${card.url}"`))
  assert.match(html, /Q &amp; K: &lt;precision> &quot;matters&quot;/)
  assert(!html.includes("Written: September"))
  assert(!html.includes("og:width"))
  assert(!html.includes("static/og-image.png"))
  const noBase = render(<>{component({ ...props, cfg: { ...cfg, baseUrl: undefined } })}</>)
  assert(!noBase.includes('property="og:image"'))
  assert(!noBase.includes('rel="canonical"'))
})

test("frontmatter cover assets reach the renderer and changing the pixels changes the card URL", async () => {
  const directory = await fs.mkdtemp(path.join(os.tmpdir(), "nuggets-cover-"))
  const output = path.join(directory, "output")
  const ctx = { argv: { directory, output }, cfg: { configuration: cfg } } as BuildCtx
  const process = async (source: string) => {
    const file = new VFile({
      path: path.join(directory, "nested/note.md"),
      value: `---\ntitle: A cover\nsocialImage: ${JSON.stringify(source)}\nsocialImageAlt: An actual article illustration.\n---\nThe body stays unchanged.`,
    })
    const processor = unified()
      .use(remarkParse)
      .use(FrontMatter().markdownPlugins!(ctx))
      .use(SocialImage().markdownPlugins!(ctx))
    await processor.run(processor.parse(file), file)
    return file
  }
  try {
    await fs.mkdir(path.join(directory, "media"))
    const cover = path.join(directory, "media/cover.png")
    await fs.copyFile("quartz/static/og-image.png", cover)
    const before = await process("media/cover.png")
    assert.equal(before.data.socialImage?.sourcePath, cover)
    assert(before.data.socialImage?.dataUrl.startsWith("data:image/png;base64,"))
    assert.match(socialMetadata(cfg, before.data).imageAlt, /An actual article illustration/)
    const rendered = await writeSocialImage(ctx, cfg, before.data)
    const png = await fs.readFile(rendered)
    assert.equal(png.readUInt32BE(16), 1200)
    assert.equal(png.readUInt32BE(20), 630)
    await fs.copyFile("content/media/social/kda-reference.png", cover)
    const after = await process("media/cover.png")
    assert.notEqual(
      socialMetadata(cfg, after.data).imageSlug,
      socialMetadata(cfg, before.data).imageSlug,
    )
    for (const source of [
      "../secret.png",
      "https://example.com/cover.png",
      "media/../cover.png",
      "media/cover.svg",
      "media/cover.png?x=1",
    ]) {
      await assert.rejects(process(source), /note.md: socialImage must/)
    }
    await assert.rejects(process("media/missing.png"), /note.md:.*ENOENT/)
    await fs.symlink(
      path.resolve("quartz/static/og-image.png"),
      path.join(directory, "media/escaped.png"),
    )
    await assert.rejects(process("media/escaped.png"), /symlink escapes/)
  } finally {
    await fs.rm(directory, { recursive: true, force: true })
  }
})

test("the production image writer emits valid, compact PNGs for each illustration and long titles", async () => {
  const output = await fs.mkdtemp(path.join(os.tmpdir(), "nuggets-social-"))
  const ctx = { argv: { output } } as BuildCtx
  try {
    const pages = [
      page(),
      page("index", "Welcome"),
      page("numerics", "Why Rtol and Atol are set the way they are in pytorch assert_close?"),
    ]
    const images: Buffer[] = []
    for (const file of pages) {
      const destination = await writeSocialImage(ctx, cfg, file)
      assert.equal(destination, path.join(output, `${socialMetadata(cfg, file).imageSlug}.png`))
      const png = await fs.readFile(destination)
      assert.equal(png.subarray(0, 8).toString("hex"), "89504e470d0a1a0a")
      assert.equal(png.readUInt32BE(16), SOCIAL_WIDTH)
      assert.equal(png.readUInt32BE(20), SOCIAL_HEIGHT)
      assert(png.length > 10_000 && png.length < 1_000_000)
      images.push(png)
    }
    assert(!images[0].equals(images[1]))
    assert(!images[1].equals(images[2]))
  } finally {
    await fs.rm(output, { recursive: true, force: true })
  }
})
