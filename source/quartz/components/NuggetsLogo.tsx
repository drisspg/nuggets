import { joinSegments, pathToRoot } from "../util/path"
import { QuartzComponent, QuartzComponentConstructor, QuartzComponentProps } from "./types"

const NuggetsLogo: QuartzComponent = ({ fileData }: QuartzComponentProps) => {
  const baseDir = pathToRoot(fileData.slug!)
  const iconPath = joinSegments(baseDir, "static/icon.svg")

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
}
`

export default (() => NuggetsLogo) satisfies QuartzComponentConstructor
