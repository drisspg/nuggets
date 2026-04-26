import { PageLayout, SharedLayout } from "./quartz/cfg"
import * as Component from "./quartz/components"
import { SimpleSlug } from "./quartz/util/path"

// components shared across all pages
export const sharedPageComponents: SharedLayout = {
  head: Component.Head(),
  header: [Component.PublicationNav(), Component.Search(), Component.Darkmode()],
  afterBody: [Component.TagList()],
  footer: Component.Footer({
    links: {
      Source: "https://github.com/drisspg/nuggets",
      GitHub: "https://github.com/drisspg",
    },
  }),
}

// components for pages that display a single page (e.g. a single note)
export const defaultContentPageLayout: PageLayout = {
  beforeBody: [
    Component.ArticleHeader(),
    Component.RecentNotes({
      limit: 5,
      linkToMore: "All-Notes" as SimpleSlug,
      linkText: "All Notes",
      showTags: false,
      showIf: (page) => page.slug === "index",
      filter: (page) =>
        Boolean(page.dates) &&
        page.slug !== "index" &&
        page.slug !== "All-Notes" &&
        !page.slug?.startsWith("tags/"),
    }),
  ],
  left: [],
  right: [Component.DesktopOnly(Component.TableOfContents())],
}

// components for pages that display lists of pages  (e.g. tags or folders)
export const defaultListPageLayout: PageLayout = {
  beforeBody: [Component.ArticleHeader()],
  left: [],
  right: [],
}
