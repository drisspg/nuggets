import { joinSegments, pathToRoot } from "../util/path"
import { QuartzComponent, QuartzComponentConstructor, QuartzComponentProps } from "./types"

const NuggetsLogo: QuartzComponent = ({ fileData }: QuartzComponentProps) => {
  const baseDir = pathToRoot(fileData.slug!)
  const iconPath = joinSegments(baseDir, "static/icon.png")

  return (
    <div class="nuggets-logo">
      <img src={iconPath} alt="Nuggets logo" />
    </div>
  )
}

NuggetsLogo.css = `
.nuggets-logo {
  display: flex;
  justify-content: center;
  margin: 1rem 0;
}

.nuggets-logo img {
  width: 96px;
  height: auto;
  border-radius: 0.75rem;
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
}
`

export default (() => NuggetsLogo) satisfies QuartzComponentConstructor
