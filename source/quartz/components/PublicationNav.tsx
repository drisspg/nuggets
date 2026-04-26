import { joinSegments, pathToRoot } from "../util/path"
import { classNames } from "../util/lang"
import { QuartzComponent, QuartzComponentConstructor, QuartzComponentProps } from "./types"

const PublicationNav: QuartzComponent = ({ cfg, fileData, displayClass }: QuartzComponentProps) => {
  const baseDir = pathToRoot(fileData.slug!)
  const iconPath = joinSegments(baseDir, "static/icon.png")

  return (
    <nav class={classNames(displayClass, "publication-nav")} aria-label="Publication">
      <a class="publication-mark" href={baseDir}>
        <span>{cfg.pageTitle}</span>
        <img src={iconPath} alt="" aria-hidden="true" />
      </a>
      <div class="publication-links">
        <a class="publication-link-pill" href={baseDir}>
          Home
        </a>
        <a
          class="publication-link-pill publication-link-featured"
          href={joinSegments(baseDir, "All-Notes")}
        >
          All Notes
        </a>
        <a href="https://github.com/drisspg">GitHub</a>
        <a href="https://x.com/drisspg">Twitter</a>
      </div>
    </nav>
  )
}

PublicationNav.css = `
.publication-nav {
  align-items: center;
  border-bottom: 1px solid var(--editorial-rule, var(--lightgray));
  display: flex;
  flex: 1 1 auto;
  font-family: var(--codeFont);
  gap: 1rem;
  justify-content: space-between;
  padding-bottom: 0.95rem;
}

.publication-mark {
  align-items: center;
  color: var(--dark);
  display: inline-flex;
  font-family: var(--headerFont);
  font-size: clamp(1.65rem, 4vw, 2.7rem);
  font-weight: 600;
  gap: 0.45rem;
  letter-spacing: -0.07em;
  line-height: 0.9;
}

.publication-mark img {
  border: 1px solid var(--editorial-rule, var(--lightgray));
  border-radius: 8px;
  box-shadow: var(--editorial-shadow, 0 4px 12px rgba(0, 0, 0, 0.15));
  height: clamp(1.9rem, 4vw, 2.6rem);
  width: auto;
}

.publication-mark:hover {
  color: var(--dark) !important;
}

.publication-links {
  align-items: center;
  display: flex;
  flex-wrap: wrap;
  gap: 0.75rem;
  justify-content: flex-end;
}

.publication-links a {
  color: var(--secondary);
  font-size: 0.68rem;
  letter-spacing: 0.13em;
  text-transform: uppercase;
}

.publication-links a:hover {
  color: var(--dark) !important;
}

.publication-links a.publication-link-pill {
  border: 1px solid var(--editorial-rule, var(--lightgray));
  border-radius: 999px;
  padding: 0.28rem 0.58rem;
}

.publication-links a.publication-link-pill:hover {
  border-color: var(--secondary);
}

.publication-links a.publication-link-featured {
  background: var(--highlight);
  border-color: color-mix(in srgb, var(--secondary) 45%, var(--editorial-rule, var(--lightgray)));
}

@media all and (max-width: 800px) {
  .publication-nav {
    align-items: flex-start;
    flex-direction: column;
  }

  .publication-links {
    justify-content: flex-start;
  }
}
`

export default (() => PublicationNav) satisfies QuartzComponentConstructor
