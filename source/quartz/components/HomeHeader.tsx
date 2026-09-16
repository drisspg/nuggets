import { FullSlug, resolveRelative } from "../util/path"
import { QuartzComponent, QuartzComponentConstructor } from "./types"
import style from "./styles/homeHeader.scss"
// @ts-ignore
import script from "./scripts/chargeBackground.inline"

const FIELD_RADIUS = 240

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
        <defs>
          {["warm", "cool"].map((tone) => (
            <radialGradient
              id={`charge-${tone}-field`}
              class={`charge-gradient-${tone}`}
              gradientUnits="userSpaceOnUse"
              cx="0"
              cy="0"
              r={FIELD_RADIUS}
            >
              <stop offset="0" class="packet-highlight" stop-opacity="0.32" />
              <stop offset="0.32" class="packet-color" stop-opacity="0.23" />
              <stop offset="0.68" class="packet-edge" stop-opacity="0.1" />
              <stop offset="1" class="packet-edge" stop-opacity="0" />
            </radialGradient>
          ))}
          <g id="charge-packet-template">
            <circle class="charge-field" r={FIELD_RADIUS} />
          </g>
        </defs>
      </svg>
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
