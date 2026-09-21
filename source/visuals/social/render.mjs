import fs from "node:fs/promises"
import path from "node:path"
import { fileURLToPath } from "node:url"
import { Resvg } from "@resvg/resvg-js"

const source = path.join(path.dirname(fileURLToPath(import.meta.url)), "covers")
const root = path.resolve(source, "../../..")
const output = path.join(root, "content/media/social")
await fs.mkdir(output, { recursive: true })
for (const name of (await fs.readdir(source)).filter((name) => name.endsWith(".svg"))) {
  let svg = await fs.readFile(path.join(source, name), "utf8")
  for (const [, href] of svg.matchAll(/href="([^"]+)"/g)) {
    if (href.startsWith("data:")) continue
    const bytes = await fs.readFile(path.resolve(source, href))
    svg = svg.replaceAll(
      `href="${href}"`,
      `href="data:image/png;base64,${bytes.toString("base64")}"`,
    )
  }
  const png = new Resvg(svg, {
    font: {
      loadSystemFonts: false,
      fontFiles: ["IBMPlexSans-Regular.ttf", "IBMPlexSans-SemiBold.ttf"].map((name) =>
        path.join(root, "quartz/fonts", name),
      ),
      defaultFontFamily: "IBM Plex Sans",
    },
  })
    .render()
    .asPng()
  const destination = path.join(output, name.replace(/\.svg$/, ".png"))
  await fs.writeFile(destination, png)
  console.log(destination)
}
