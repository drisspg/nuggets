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
        header: "Newsreader",
        body: "IBM Plex Sans",
        code: "IBM Plex Mono",
      },
      colors: {
        lightMode: {
          light: "#faeadc",
          lightgray: "#ddd6c8",
          gray: "#918a7c",
          darkgray: "#354139",
          dark: "#18221d",
          secondary: "#5f7f67",
          tertiary: "#7f8f86",
          highlight: "rgba(95, 127, 103, 0.12)",
          textHighlight: "rgba(127, 143, 134, 0.24)",
        },
        darkMode: {
          light: "#0e0e0e",
          lightgray: "#1f211d",
          gray: "#7a7d76",
          darkgray: "#c8c9c4",
          dark: "#e8e9e4",
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
      Plugin.SyntaxHighlighting({
        theme: {
          light: "github-light",
          dark: "catppuccin-mocha",
        },
        keepBackground: false,
      }),
      Plugin.ObsidianFlavoredMarkdown({ enableInHtmlEmbed: true }),
      Plugin.GitHubFlavoredMarkdown(),
      Plugin.TableOfContents(),
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
