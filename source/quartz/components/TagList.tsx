import { pathToRoot, slugTag } from "../util/path"
import { QuartzComponent, QuartzComponentConstructor, QuartzComponentProps } from "./types"
import { classNames } from "../util/lang"

function displayTag(tag: string): string {
  return tag
    .split("-")
    .map((part) => part.slice(0, 1).toUpperCase() + part.slice(1))
    .join(" ")
}

const TagList: QuartzComponent = ({ fileData, displayClass }: QuartzComponentProps) => {
  const tags = fileData.frontmatter?.tags
  const baseDir = pathToRoot(fileData.slug!)
  if (tags && tags.length > 0) {
    return (
      <nav class={classNames(displayClass, "tag-footer")} aria-label="Post tags">
        <span>Tagged</span>
        <ul class="tags">
          {tags.map((tag) => {
            const linkDest = baseDir + `/tags/${slugTag(tag)}`
            return (
              <li>
                <a href={linkDest} class="internal tag-link">
                  {displayTag(tag)}
                </a>
              </li>
            )
          })}
        </ul>
      </nav>
    )
  } else {
    return null
  }
}

TagList.css = `
.tag-footer {
  align-items: baseline;
  display: flex;
  flex-wrap: wrap;
  gap: 0.6rem;
  margin: 0.5rem 0 0;
}

.tag-footer > span {
  color: var(--darkgray);
  font-family: var(--codeFont);
  font-size: 0.72rem;
  letter-spacing: 0.12em;
  text-transform: uppercase;
}

.tags {
  list-style: none;
  display: flex;
  padding-left: 0;
  gap: 0.4rem;
  margin: 0;
  flex-wrap: wrap;
}

.section-li > .section > .tags {
  justify-content: flex-end;
}
  
.tags > li {
  display: inline-block;
  white-space: nowrap;
  margin: 0;
  overflow-wrap: normal;
}

a.internal.tag-link {
  border-radius: 999px;
  background-color: var(--highlight);
  padding: 0.2rem 0.48rem;
  margin: 0 0.1rem;
}
`

export default (() => TagList) satisfies QuartzComponentConstructor
