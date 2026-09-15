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
        <a class="publication-social" href="https://github.com/drisspg">
          <svg viewBox="0 0 24 24" aria-hidden="true" focusable="false">
            <path d="M8 21v-4c-3-.5-5-2-5-5 0-1.5.5-2.7 1.5-3.7C4.2 7.5 4.3 6.3 4.8 5c1.6 0 2.8.7 3.6 1.3a12 12 0 0 1 7.2 0C16.4 5.7 17.6 5 19.2 5c.5 1.3.6 2.5.3 3.3C20.5 9.3 21 10.5 21 12c0 3-2 4.5-5 5v4M8 19c-3 1-4-1-5-2" />
          </svg>
          <span>GitHub</span>
        </a>
        <a class="publication-social" href="https://x.com/drisspg">
          <svg viewBox="0 0 24 24" aria-hidden="true" focusable="false">
            <path d="M5 4h4l10 16h-4L5 4ZM19 4 5 20" />
          </svg>
          <span>Twitter</span>
        </a>
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

.publication-links .publication-social {
  align-items: center;
  background: color-mix(in srgb, var(--secondary) 5%, transparent);
  border: 1px solid var(--editorial-rule, var(--lightgray));
  border-radius: 6px;
  display: inline-flex;
  gap: 0.45rem;
  padding: 0.4rem 0.6rem;
  line-height: 1.2;
}

.publication-social svg {
  fill: none;
  flex-shrink: 0;
  height: 15px;
  width: 15px;
  stroke: currentColor;
  stroke-width: 1.5;
  stroke-linecap: round;
  stroke-linejoin: round;
}

.publication-links .publication-social:hover {
  background: color-mix(in srgb, var(--secondary) 12%, transparent);
  border-color: var(--secondary);
}

.publication-links a:focus-visible {
  outline: 2px solid var(--secondary);
  outline-offset: 4px;
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
