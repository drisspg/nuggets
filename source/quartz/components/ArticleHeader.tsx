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
  border-bottom: 1px solid var(--editorial-rule, var(--lightgray));
  margin: 1.25rem 0 2.35rem;
  padding-bottom: 1.65rem;
}

.article-hero h1 {
  color: var(--dark);
  font-family: var(--headerFont);
  font-size: clamp(2.6rem, 5vw, 4.6rem);
  font-weight: 600;
  letter-spacing: -0.06em;
  line-height: 0.98;
  margin: 0;
  max-width: 24ch;
}

.article-dek {
  color: var(--darkgray);
  font-size: clamp(1.08rem, 1.8vw, 1.3rem);
  line-height: 1.55;
  margin: 1.1rem 0 0;
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

@media all and (max-width: 800px) {
  .article-hero {
    display: block;
  }

  .article-hero h1 {
    font-size: clamp(3rem, 18vw, 5.5rem);
    max-width: none;
  }


  .article-dek {
    margin-top: 1rem;
  }
}
`

export default (() => ArticleHeader) satisfies QuartzComponentConstructor
