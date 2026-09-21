import { i18n } from "../i18n"
import { FullSlug, joinSegments, pathToRoot } from "../util/path"
import { JSResourceToScriptElement } from "../util/resources"
import { googleFontHref } from "../util/theme"
import { SOCIAL_HEIGHT, SOCIAL_WIDTH, socialMetadata } from "../util/social"
import { QuartzComponent, QuartzComponentConstructor, QuartzComponentProps } from "./types"

export default (() => {
  const Head: QuartzComponent = ({ cfg, fileData, externalResources }: QuartzComponentProps) => {
    const titleSuffix = cfg.pageTitleSuffix ?? ""
    const title =
      (fileData.frontmatter?.title ?? i18n(cfg.locale).propertyDefaults.title) + titleSuffix
    const social = socialMetadata(cfg, fileData)
    const description = social.description
    const { css, js } = externalResources

    const url = new URL(`https://${cfg.baseUrl ?? "example.com"}`)
    const path = url.pathname as FullSlug
    const baseDir = fileData.slug === "404" ? path : pathToRoot(fileData.slug!)

    const iconPath = joinSegments(baseDir, "static/icon.svg")

    return (
      <head>
        <title>{title}</title>
        <meta charSet="utf-8" />
        {cfg.theme.cdnCaching && cfg.theme.fontOrigin === "googleFonts" && (
          <>
            <link rel="preconnect" href="https://fonts.googleapis.com" />
            <link rel="preconnect" href="https://fonts.gstatic.com" />
            <link rel="stylesheet" href={googleFontHref(cfg.theme)} />
          </>
        )}
        <meta name="viewport" content="width=device-width, initial-scale=1.0" />
        <meta property="og:title" content={social.title} />
        <meta property="og:description" content={description} />
        <meta property="og:site_name" content={cfg.pageTitle} />
        <meta property="og:type" content={social.type} />
        {social.url && <meta property="og:url" content={social.url} />}
        {social.url && <link rel="canonical" href={social.url} />}
        <meta name="twitter:card" content="summary_large_image" />
        <meta name="twitter:creator" content="@drisspg" />
        <meta name="twitter:title" content={social.title} />
        <meta name="twitter:description" content={description} />
        {social.imageUrl && (
          <>
            <meta property="og:image" content={social.imageUrl} />
            <meta property="og:image:type" content="image/png" />
            <meta property="og:image:width" content={String(SOCIAL_WIDTH)} />
            <meta property="og:image:height" content={String(SOCIAL_HEIGHT)} />
            <meta property="og:image:alt" content={social.imageAlt} />
            <meta name="twitter:image" content={social.imageUrl} />
            <meta name="twitter:image:alt" content={social.imageAlt} />
          </>
        )}
        <link rel="icon" type="image/svg+xml" href={iconPath} />
        <meta name="description" content={description} />
        {css.map((href) => (
          <link key={href} href={href} rel="stylesheet" type="text/css" spa-preserve />
        ))}
        {js
          .filter((resource) => resource.loadTime === "beforeDOMReady")
          .map((res) => JSResourceToScriptElement(res, true))}
      </head>
    )
  }

  return Head
}) satisfies QuartzComponentConstructor
