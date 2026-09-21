import { createHash } from "node:crypto"
import { GlobalConfiguration } from "../cfg"
import { QuartzPluginData } from "../plugins/vfile"
import { FullSlug, simplifySlug } from "./path"

export const SOCIAL_WIDTH = 1200
export const SOCIAL_HEIGHT = 630

export function socialMetadata(cfg: GlobalConfiguration, file: QuartzPluginData) {
  const slug = file.slug ?? ("index" as FullSlug)
  const home = slug === "index"
  const title = home ? cfg.pageTitle : (file.frontmatter?.title ?? cfg.pageTitle)
  const description = (
    file.frontmatter?.description ??
    (typeof file.frontmatter?.dek === "string" ? file.frontmatter.dek : undefined) ??
    file.description ??
    "Notes on PyTorch, GPU kernels, and experiments by Driss."
  )
    .replace(/\s+/g, " ")
    .trim()
  const tags = file.frontmatter?.tags ?? []
  const collection =
    home || slug === "All-Notes" || slug.startsWith("tags/") || slug.endsWith("/index")
  const section = collection
    ? "FIELD NOTES"
    : tags.includes("pytorch")
      ? "PYTORCH / GPU ENGINEERING"
      : tags.includes("ai")
        ? "AI / EXPERIMENTS"
        : tags.includes("tool-tips")
          ? "TOOLS / PRACTICE"
          : "NOTES / OFF THE CLOCK"
  const artwork = /kda|attention/i.test(title)
    ? "attention"
    : collection || tags.includes("ai")
      ? "charges"
      : "tiles"
  const card = { title, description, section, artwork, site: cfg.pageTitle }
  // Version the design as well as the copy so shared URLs don't retain stale preview images.
  const hash = createHash("sha256")
    .update(JSON.stringify({ version: 3, ...card, coverHash: file.socialImage?.hash }))
    .digest("hex")
    .slice(0, 16)
  const imageSlug = `static/social/${hash}` as FullSlug
  const base = cfg.baseUrl ? `https://${cfg.baseUrl.replace(/\/$/, "")}/` : undefined
  return {
    ...card,
    cover: file.socialImage,
    imageSlug,
    imageUrl: base ? new URL(`${imageSlug}.png`, base).href : undefined,
    url: base ? new URL(encodeURI(simplifySlug(slug)).replace(/^\/+/, ""), base).href : undefined,
    type: collection || slug === "404" ? "website" : "article",
    imageAlt: `${title} — ${cfg.pageTitle}, notes by Driss. ${file.socialImage?.alt ?? (artwork === "attention" ? "A token diagram with green past connections and red future connections around an orange reference." : "An abstract GPU-inspired illustration in green and amber.")}`,
  }
}

export type SocialMetadata = ReturnType<typeof socialMetadata>
