import { createHash } from "node:crypto"
import { readFileSync, realpathSync } from "node:fs"
import path from "node:path"
import { QuartzTransformerPlugin } from "../types"
import { FilePath } from "../../util/path"

export const SocialImage: QuartzTransformerPlugin = () => ({
  name: "SocialImage",
  markdownPlugins(ctx) {
    return [
      () => (_tree, file) => {
        const source = file.data.frontmatter?.socialImage
        if (source === undefined) return
        try {
          if (
            typeof source !== "string" ||
            !source.startsWith("media/") ||
            /[\\?#%\u0000-\u001f\u007f]/.test(source) ||
            source.split("/").some((part) => !part || part === "." || part === "..") ||
            ![".png", ".jpg", ".jpeg", ".webp"].includes(path.extname(source))
          )
            throw new Error("socialImage must be a PNG, JPEG, or WebP path within media/")
          const root = realpathSync(ctx.argv.directory)
          const sourcePath = realpathSync(path.resolve(root, source))
          const relative = path.relative(root, sourcePath)
          if (
            relative === ".." ||
            relative.startsWith(`..${path.sep}`) ||
            path.isAbsolute(relative)
          ) {
            throw new Error("socialImage symlink escapes the content root")
          }
          const bytes = readFileSync(sourcePath)
          const type = source.endsWith(".png") ? "png" : source.endsWith(".webp") ? "webp" : "jpeg"
          file.data.socialImage = {
            sourcePath: path.join(ctx.argv.directory, source) as FilePath,
            dataUrl: `data:image/${type};base64,${bytes.toString("base64")}`,
            hash: createHash("sha256").update(bytes).digest("hex"),
            alt:
              typeof file.data.frontmatter?.socialImageAlt === "string"
                ? file.data.frontmatter.socialImageAlt
                : `Cover illustration for ${file.data.frontmatter?.title}`,
          }
        } catch (error) {
          throw new Error(`${file.path}: ${error instanceof Error ? error.message : error}`, {
            cause: error,
          })
        }
      },
    ]
  },
})

declare module "vfile" {
  interface DataMap {
    socialImage: { sourcePath: FilePath; dataUrl: string; hash: string; alt: string }
  }
}
