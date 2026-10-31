// Link inline code in prose to the notebook lines it quotes. Prose usually discusses
// the cell just above it, so that cell is searched first, then the one below.
const minimumQuote = 3
// A quote that matches more lines than this is a generic name, not a pointer.
const maximumHits = 24

const normalize = (text: string) => text.replace(/\s+/g, " ").trim()

function cellLines(cell: HTMLElement) {
  return Array.from(cell.querySelectorAll<HTMLElement>("pre [data-line]")).map((line) => ({
    line,
    text: normalize(line.textContent ?? ""),
  }))
}

function revealWithinBox(line: HTMLElement) {
  const box = line.closest("pre")
  if (!box || box.scrollHeight <= box.clientHeight) return
  box.scrollTo({ top: line.offsetTop - box.clientHeight / 3, behavior: "smooth" })
}

function linkNotebookProse(article: HTMLElement) {
  const cells = Array.from(article.querySelectorAll<HTMLElement>(".nb-cell"))
  if (cells.length === 0) return
  const lines = new Map(cells.map((cell) => [cell, cellLines(cell)]))
  const controller = new AbortController()
  const { signal } = controller
  const linked: HTMLElement[] = []
  let pinned: HTMLElement | undefined

  const quotes = Array.from(article.querySelectorAll<HTMLElement>("code")).filter(
    (code) => !code.closest("pre, .nb-cell, a"),
  )
  for (const quote of quotes) {
    const needle = normalize(quote.textContent ?? "")
    if (needle.length < minimumQuote) continue
    const following = cells.findIndex(
      (cell) => quote.compareDocumentPosition(cell) & Node.DOCUMENT_POSITION_FOLLOWING,
    )
    const below = following === -1 ? undefined : cells[following]
    const above = following === -1 ? cells.at(-1) : cells[following - 1]
    let hits: HTMLElement[] = []
    for (const cell of [above, below]) {
      if (!cell) continue
      hits = lines
        .get(cell)!
        .filter(({ text }) => text.includes(needle))
        .map(({ line }) => line)
      if (hits.length > 0) break
    }
    if (hits.length === 0 || hits.length > maximumHits) continue

    const cell = hits[0].closest<HTMLElement>(".nb-cell")!
    const show = () => {
      for (const line of hits) line.dataset.nbHit = ""
      cell.dataset.nbReferenced = ""
      for (const box of new Set(hits.map((line) => line.closest("pre")))) {
        revealWithinBox(hits.find((line) => line.closest("pre") === box)!)
      }
    }
    const hide = () => {
      if (pinned === quote) return
      for (const line of hits) delete line.dataset.nbHit
      delete cell.dataset.nbReferenced
    }
    quote.classList.add("nb-quote")
    quote.tabIndex = 0
    quote.title = "Highlight in the notebook cell"
    linked.push(quote)
    quote.addEventListener("pointerenter", show, { signal })
    quote.addEventListener("focus", show, { signal })
    quote.addEventListener("pointerleave", hide, { signal })
    quote.addEventListener("blur", hide, { signal })
    quote.addEventListener(
      "click",
      () => {
        const previous = pinned
        pinned = previous === quote ? undefined : quote
        previous?.classList.remove("nb-quote-pinned")
        previous?.dispatchEvent(new Event("pointerleave"))
        if (pinned) {
          quote.classList.add("nb-quote-pinned")
          show()
          const rect = cell.getBoundingClientRect()
          if (rect.bottom < 0 || rect.top > window.innerHeight) {
            cell.scrollIntoView({ behavior: "smooth", block: "center" })
          }
        }
      },
      { signal },
    )
  }

  window.addCleanup(() => {
    controller.abort()
    for (const quote of linked) {
      quote.classList.remove("nb-quote", "nb-quote-pinned")
      quote.removeAttribute("tabindex")
      quote.removeAttribute("title")
    }
    article.querySelectorAll<HTMLElement>("[data-nb-hit]").forEach((line) => {
      delete line.dataset.nbHit
    })
  })
}

document.addEventListener("nav", () => {
  const article = document.querySelector<HTMLElement>("article")
  if (article) linkNotebookProse(article)
})
