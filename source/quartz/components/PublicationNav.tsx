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
        <a href={baseDir}>Index</a>
        <a href="https://github.com/drisspg/nuggets">Source</a>
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
