import { FilePath, pathToRoot, slugifyFilePath } from "../util/path"
import { classNames } from "../util/lang"
import { QuartzComponent, QuartzComponentConstructor, QuartzComponentProps } from "./types"

function asString(value: unknown): string | undefined {
  return typeof value === "string" && value.length > 0 ? value : undefined
}

const ArticleHeader: QuartzComponent = ({ fileData, displayClass }: QuartzComponentProps) => {
  const title = asString(fileData.frontmatter?.title) ?? "Untitled"
  const dek = asString(fileData.frontmatter?.dek) ?? asString(fileData.frontmatter?.description)
  const hero = asString(fileData.frontmatter?.hero)
  const baseDir = pathToRoot(fileData.slug!)

  return (
    <header class={classNames(displayClass, "article-hero")}>
      <h1>{title}</h1>
      {dek && <p class="article-dek">{dek}</p>}
      {hero && (
        <figure class="article-hero-media">
          <img src={baseDir + slugifyFilePath(hero as FilePath)} alt="" />
        </figure>
      )}
    </header>
  )
}

ArticleHeader.css = `
.article-hero {
  display: block;
  margin: 2.5rem 0 1.5rem;
}

.article-hero h1 {
  color: var(--heading-color, var(--dark));
  font-family: var(--headerFont);
  font-size: 2rem;
  font-weight: 700;
  letter-spacing: -0.02em;
  line-height: 1.3;
  margin: 0;
}

.article-dek {
  color: var(--darkgray);
  font-size: 1rem;
  line-height: 1.6;
  margin: 0.75rem 0 0;
  max-width: 42rem;
}


.article-hero-media {
  margin: 1.6rem 0 0;
}

.article-hero-media img {
  aspect-ratio: 16 / 9;
  border-radius: 2px;
  object-fit: cover;
  width: 100%;
}

`

export default (() => ArticleHeader) satisfies QuartzComponentConstructor
