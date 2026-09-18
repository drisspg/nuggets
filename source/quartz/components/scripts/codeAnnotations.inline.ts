function enhanceCodeAnnotations(wrapper: HTMLElement) {
  // Without native popovers, keep the linked, fully visible Markdown notes.
  if (typeof HTMLElement.prototype.showPopover !== "function") return
  const notes = wrapper.querySelector<HTMLOListElement>(":scope > .code-annotation-notes")!
  const items = Array.from(notes.children) as HTMLLIElement[]
  const anchors = Array.from(
    wrapper.querySelectorAll<HTMLAnchorElement>(".code-annotation-marker"),
  ).filter((anchor) => anchor.closest(".code-annotations") === wrapper)
  const controller = new AbortController()
  const { signal } = controller
  const panel = document.createElement("div")
  panel.className = "code-annotation-popover"
  panel.id = `${items[0].id}-popover`
  panel.popover = "auto"
  panel.tabIndex = -1
  panel.setAttribute("role", "dialog")
  const heading = document.createElement("p")
  heading.className = "code-annotation-heading"
  const close = document.createElement("button")
  close.className = "code-annotation-close"
  close.type = "button"
  close.textContent = "Close"
  close.setAttribute("aria-label", "Close code annotation")
  panel.append(heading, notes, close)
  wrapper.append(panel)
  let active: HTMLButtonElement | undefined

  const buttons = anchors.map((anchor) => {
    const button = document.createElement("button")
    button.type = "button"
    button.id = anchor.id
    button.popoverTargetElement = panel
    button.className = anchor.className
    button.dataset.codeNote = anchor.dataset.codeNote
    button.textContent = anchor.textContent
    button.setAttribute("aria-label", anchor.getAttribute("aria-label")!)
    button.setAttribute("aria-controls", panel.id)
    button.setAttribute("aria-haspopup", "dialog")
    button.setAttribute("aria-expanded", "false")
    anchor.replaceWith(button)
    return button
  })

  function clearSelection() {
    buttons.forEach((button) => {
      button.setAttribute("aria-expanded", "false")
      button.closest("[data-line]")?.removeAttribute("data-code-active")
    })
  }

  function position() {
    if (!active || !panel.matches(":popover-open")) return
    const target = active.getBoundingClientRect()
    const bounds = panel.getBoundingClientRect()
    const pre = active.closest("pre")!.getBoundingClientRect()
    if (
      target.bottom < 0 ||
      target.top > innerHeight ||
      target.right < pre.left ||
      target.left > pre.right
    ) {
      panel.hidePopover()
      return
    }
    const left = Math.max(12, Math.min(target.left, innerWidth - bounds.width - 12))
    const below = target.bottom + 8
    const top = below + bounds.height <= innerHeight - 12 ? below : target.top - bounds.height - 8
    panel.style.left = `${left}px`
    panel.style.top = `${Math.max(12, Math.min(top, innerHeight - bounds.height - 12))}px`
  }

  function dismiss(restoreFocus: boolean) {
    if (!panel.matches(":popover-open")) return
    panel.hidePopover()
    clearSelection()
    if (restoreFocus) active?.focus({ preventScroll: true })
  }

  buttons.forEach((button) => {
    button.addEventListener(
      "click",
      (event) => {
        event.preventDefault()
        if (active === button && panel.matches(":popover-open")) {
          dismiss(true)
          return
        }
        active = button
        const number = button.dataset.codeNote!
        items.forEach((item) => {
          item.hidden = item.dataset.codeNote !== number
        })
        heading.textContent = `Annotation ${number}`
        panel.setAttribute("aria-label", `Code annotation ${number}`)
        clearSelection()
        buttons
          .filter((candidate) => candidate.dataset.codeNote === number)
          .forEach((candidate) => {
            candidate.closest("[data-line]")?.setAttribute("data-code-active", "")
          })
        button.setAttribute("aria-expanded", "true")
        if (!panel.matches(":popover-open")) panel.showPopover()
        position()
        panel.focus({ preventScroll: true })
      },
      { signal },
    )
  })
  close.addEventListener("click", () => dismiss(true), { signal })
  panel.addEventListener(
    "toggle",
    () => {
      if (!panel.matches(":popover-open")) clearSelection()
    },
    { signal },
  )
  document.addEventListener(
    "keydown",
    (event) => {
      if (
        event.key === "Escape" &&
        panel.matches(":popover-open") &&
        document.activeElement?.closest(".code-annotation-popover") === panel
      ) {
        event.preventDefault()
        event.stopPropagation()
        dismiss(true)
      }
    },
    { signal, capture: true },
  )
  // Capture includes horizontal scrolling inside the code block.
  document.addEventListener("scroll", position, { signal, capture: true, passive: true })
  window.addEventListener("resize", position, { signal, passive: true })

  const observer = new ResizeObserver(position)
  observer.observe(panel)
  window.addCleanup(() => {
    dismiss(false)
    controller.abort()
    observer.disconnect()
    clearSelection()
    buttons.forEach((button, index) => button.replaceWith(anchors[index]))
    items.forEach((item) => {
      item.hidden = false
    })
    wrapper.append(notes)
    panel.remove()
  })
}

document.addEventListener("nav", () => {
  document.querySelectorAll<HTMLElement>(".code-annotations").forEach(enhanceCodeAnnotations)
})
