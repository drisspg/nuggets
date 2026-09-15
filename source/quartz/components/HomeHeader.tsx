import { FullSlug, resolveRelative } from "../util/path"
import { QuartzComponent, QuartzComponentConstructor } from "./types"
import style from "./styles/homeHeader.scss"
// @ts-ignore
import script from "./scripts/chargeBackground.inline"

const HomeHeader: QuartzComponent = ({ fileData }) => {
  const currentSlug = fileData.slug
  if (currentSlug !== "index") return null
  const dek = fileData.frontmatter?.dek

  return (
    <section class="home-intro" aria-label="Welcome to Nuggets">
      <svg
        class="home-collision"
        viewBox="0 0 600 230"
        preserveAspectRatio="xMidYMid slice"
        aria-hidden="true"
      >
        {["a", "b"].map((source) => (
          <g
            class={`charge-packet-${source}`}
            transform={`translate(${source === "a" ? 250 : 530} 115)`}
          >
            {/* Gaussian-windowed contours are illustrative, not electron wavefunctions. */}
            {Array.from({ length: 18 }, (_, i) => {
              const radius = (i + 1) * 7
              return <circle r={radius} opacity={Math.exp(-0.5 * (radius / 55) ** 2)} />
            })}
            <circle class="charge-origin" r="2.2" />
          </g>
        ))}
      </svg>
      <p class="home-byline">
        Personal notes · <a href="https://github.com/drisspg">@drisspg</a>
      </p>
      <h1>{fileData.frontmatter?.title}</h1>
      {typeof dek === "string" && <p class="home-dek">{dek}</p>}
      <nav class="home-topics" aria-label="Browse topics">
        {[
          ["pytorch", "PyTorch"],
          ["tool-tips", "Tool tips"],
          ["ai", "AI"],
        ].map(([slug, label]) => (
          <a class="internal" href={resolveRelative(currentSlug, `tags/${slug}` as FullSlug)}>
            {label}
          </a>
        ))}
      </nav>
    </section>
  )
}

HomeHeader.css = style
HomeHeader.afterDOMLoaded = script
export default (() => HomeHeader) satisfies QuartzComponentConstructor
