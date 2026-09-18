import { QuartzConfig } from "./quartz/cfg"
import * as Plugin from "./quartz/plugins"

/**
 * Quartz 4.0 Configuration
 *
 * See https://quartz.jzhao.xyz/configuration for more information.
 */
const config: QuartzConfig = {
  configuration: {
    pageTitle: "Nuggets",
    pageTitleSuffix: "",
    enableSPA: true,
    enablePopovers: false,
    analytics: {
      provider: "goatcounter",
      websiteId: "drisspg",
    },
    locale: "en-US",
    baseUrl: "drisspg.github.io/nuggets",
    ignorePatterns: ["private", "templates", ".obsidian"],
    defaultDateType: "created",
    theme: {
      fontOrigin: "googleFonts",
      cdnCaching: true,
      typography: {
        header: "IBM Plex Sans",
        body: "IBM Plex Sans",
        code: "IBM Plex Mono",
      },
      colors: {
        lightMode: {
          light: "#f4f0e8",
          lightgray: "#d0ccc4",
          gray: "#6a7a6e",
          darkgray: "#1a2e22",
          dark: "#1a2e22",
          secondary: "#5f7f67",
          tertiary: "#7f8f86",
          highlight: "rgba(95, 127, 103, 0.12)",
          textHighlight: "rgba(127, 143, 134, 0.24)",
        },
        darkMode: {
          light: "#0e0e0e",
          lightgray: "#202020",
          gray: "#999999",
          darkgray: "#d4d4d4",
          dark: "#d4d4d4",
          secondary: "#6f8f7b",
          tertiary: "#7f8f86",
          highlight: "rgba(111, 143, 123, 0.16)",
          textHighlight: "rgba(111, 143, 123, 0.26)",
        },
      },
    },
  },
  plugins: {
    transformers: [
      Plugin.FrontMatter(),
      Plugin.CreatedModifiedDate({
        priority: ["frontmatter", "filesystem"],
      }),
      Plugin.DocEmbeds(),
      Plugin.NativeCharts(),
      Plugin.SyntaxHighlighting({
        theme: {
          light: "github-light",
          dark: "github-dark",
        },
        keepBackground: false,
      }),
      Plugin.CodeAnnotations(),
      Plugin.ObsidianFlavoredMarkdown({ enableInHtmlEmbed: true }),
      Plugin.GitHubFlavoredMarkdown(),
      Plugin.TableOfContents({ collapseByDefault: true }),
      Plugin.CrawlLinks({ markdownLinkResolution: "shortest" }),
      Plugin.Description(),
      Plugin.Latex({ renderEngine: "katex" }),
    ],
    filters: [Plugin.RemoveDrafts()],
    emitters: [
      Plugin.AliasRedirects(),
      Plugin.ComponentResources(),
      Plugin.ContentPage(),
      Plugin.FolderPage(),
      Plugin.TagPage(),
      Plugin.ContentIndex({
        enableSiteMap: true,
        enableRSS: true,
      }),
      Plugin.Assets(),
      Plugin.Static(),
      Plugin.NotFoundPage(),
    ],
  },
}

export default config
