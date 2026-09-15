import { joinSegments, pathToRoot } from "../util/path"
import { classNames } from "../util/lang"
import { QuartzComponent, QuartzComponentConstructor, QuartzComponentProps } from "./types"

const PublicationNav: QuartzComponent = ({ cfg, fileData, displayClass }: QuartzComponentProps) => {
  const baseDir = pathToRoot(fileData.slug!)
  const iconPath = joinSegments(baseDir, "static/icon.svg")

  return (
    <nav class={classNames(displayClass, "publication-nav")} aria-label="Publication">
      <a class="publication-mark" href={baseDir}>
        <span>{cfg.pageTitle}</span>
        <img src={iconPath} alt="" aria-hidden="true" width="32" height="32" />
      </a>
      <div class="publication-links">
        <a href={baseDir}>Home</a>
        <a href={joinSegments(baseDir, "All-Notes")}>All Notes</a>
        <a href="https://github.com/drisspg">GitHub</a>
        <a href="https://x.com/drisspg">Twitter</a>
      </div>
    </nav>
  )
}

PublicationNav.css = `
.publication-nav {
  align-items: center;
  display: flex;
  flex: 1 1 auto;
  font-family: var(--bodyFont);
  gap: 1rem;
  justify-content: space-between;
  min-width: 0;
}

.publication-mark {
  align-items: center;
  color: var(--dark);
  display: inline-flex;
  font-family: var(--headerFont);
  font-size: 1.125rem;
  font-weight: 700;
  gap: 0.65rem;
  line-height: 1.4;
}

.publication-mark img {
  height: 2rem;
  order: -1;
  width: 2rem;
}

.publication-mark:hover {
  color: var(--dark) !important;
}

.publication-links {
  align-items: center;
  display: flex;
  flex-wrap: wrap;
  gap: 1rem;
  justify-content: flex-end;
}

.publication-links a {
  color: var(--darkgray);
  font-size: 0.875rem;
  font-weight: 400;
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
