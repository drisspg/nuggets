// Link inline code in prose to the notebook lines it quotes. Commentary searches its own
// cell; other prose usually discusses the cell just above it, then the one below.
const minimumQuote = 3
// A quote that matches more lines than this is a generic name, not a pointer.
const maximumHits = 24

const normalize = (text: string) => text.replace(/\s+/g, " ").trim()

// Only the cell's code and log count; code blocks quoted in the prose would match themselves.
function cellLines(cell: HTMLElement) {
  return Array.from(
    cell.querySelectorAll<HTMLElement>(".nb-input pre [data-line], .nb-output pre [data-line]"),
  ).map((line) => ({
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
  let pinned: { quote: HTMLElement; show: (reveal: boolean) => void } | undefined

  const quotes = Array.from(article.querySelectorAll<HTMLElement>("code")).filter(
    (code) => !code.closest("pre, a, .nb-prompt"),
  )
  for (const quote of quotes) {
    const needle = normalize(quote.textContent ?? "")
    if (needle.length < minimumQuote) continue
    const own = quote.closest<HTMLElement>(".nb-cell")
    const following = cells.findIndex(
      (cell) => quote.compareDocumentPosition(cell) & Node.DOCUMENT_POSITION_FOLLOWING,
    )
    const below = following === -1 ? undefined : cells[following]
    const above = following === -1 ? cells.at(-1) : cells[following - 1]
    // Inductor's IR dumps print private fields as attributes: `_sizes` appears as `op0.sizes`.
    const needles = /^_\w+$/.test(needle) ? [needle, `.${needle.slice(1)}`] : [needle]
    let hits: HTMLElement[] = []
    for (const cell of own ? [own] : [above, below]) {
      if (!cell) continue
      hits = lines
        .get(cell)!
        .filter(({ text }) => needles.some((candidate) => text.includes(candidate)))
        .map(({ line }) => line)
      if (hits.length > 0) break
    }
    if (hits.length === 0 || hits.length > maximumHits) continue

    const cell = hits[0].closest<HTMLElement>(".nb-cell")!
    const show = (reveal = true) => {
      for (const line of hits) line.dataset.nbHit = ""
      cell.dataset.nbReferenced = ""
      if (!reveal) return
      for (const box of new Set(hits.map((line) => line.closest("pre")))) {
        revealWithinBox(hits.find((line) => line.closest("pre") === box)!)
      }
    }
    const clear = () => {
      for (const line of hits) delete line.dataset.nbHit
      delete cell.dataset.nbReferenced
    }
    // Hovering another quote must not erase the pinned quote's highlight.
    const hide = () => {
      if (pinned?.quote === quote) return
      clear()
      pinned?.show(false)
    }
    quote.classList.add("nb-quote")
    quote.tabIndex = 0
    quote.title = "Highlight in the notebook cell"
    linked.push(quote)
    quote.addEventListener("pointerenter", () => show(), { signal })
    quote.addEventListener("focus", () => show(), { signal })
    quote.addEventListener("pointerleave", hide, { signal })
    quote.addEventListener("blur", hide, { signal })
    quote.addEventListener(
      "click",
      () => {
        const previous = pinned
        pinned = previous?.quote === quote ? undefined : { quote, show }
        previous?.quote.classList.remove("nb-quote-pinned")
        previous?.quote.dispatchEvent(new Event("pointerleave"))
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

// Every pane can open in a floating dialog above the page, at full size.
function addFocusButtons(article: HTMLElement) {
  if (!article.querySelector(".nb-cell")) return
  const dialog = document.createElement("dialog")
  dialog.className = "nb-focus"
  dialog.setAttribute("aria-label", "Notebook cell")
  const header = document.createElement("div")
  header.className = "nb-focus-header"
  const title = document.createElement("span")
  title.className = "nb-prompt"
  const copy = document.createElement("button")
  copy.type = "button"
  copy.textContent = "copy"
  const close = document.createElement("button")
  close.type = "button"
  close.textContent = "close ✕"
  const body = document.createElement("div")
  body.className = "nb-focus-body"
  header.append(title, copy, close)
  dialog.append(header, body)
  document.body.append(dialog)

  const controller = new AbortController()
  const { signal } = controller
  close.addEventListener("click", () => dialog.close(), { signal })
  // A click on the backdrop lands on the dialog element itself.
  dialog.addEventListener("click", (event) => event.target === dialog && dialog.close(), {
    signal,
  })
  dialog.addEventListener("close", () => body.replaceChildren(), { signal })
  copy.addEventListener(
    "click",
    () => {
      const code = body.querySelector("code")
      if (!code || !navigator.clipboard) return
      navigator.clipboard.writeText(code.innerText).then(
        () => {
          copy.textContent = "copied"
          setTimeout(() => (copy.textContent = "copy"), 1200)
        },
        () => (copy.textContent = "copy failed"),
      )
    },
    { signal },
  )

  for (const pane of article.querySelectorAll<HTMLElement>(".nb-input, .nb-output")) {
    const prompt = pane.querySelector<HTMLElement>(".nb-prompt")!
    const figure = pane.querySelector<HTMLElement>("figure, pre")
    if (!figure) continue
    const button = document.createElement("button")
    button.type = "button"
    button.className = "nb-expand"
    button.textContent = "expand ⤢"
    button.setAttribute("aria-label", `Expand ${prompt.textContent}`)
    button.addEventListener(
      "click",
      () => {
        title.textContent = prompt.firstChild?.textContent ?? ""
        title.className = prompt.className
        const clone = figure.cloneNode(true) as HTMLElement
        clone
          .querySelectorAll("[data-nb-hit]")
          .forEach((line) => line.removeAttribute("data-nb-hit"))
        body.replaceChildren(clone)
        dialog.showModal()
        close.focus()
      },
      { signal },
    )
    prompt.append(button)
  }

  window.addCleanup(() => {
    controller.abort()
    if (dialog.open) dialog.close()
    dialog.remove()
    article.querySelectorAll(".nb-expand").forEach((button) => button.remove())
  })
}

document.addEventListener("nav", () => {
  const article = document.querySelector<HTMLElement>("article")
  if (!article) return
  linkNotebookProse(article)
  addFocusButtons(article)
})
