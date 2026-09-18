const svgCopy =
  '<svg aria-hidden="true" height="16" viewBox="0 0 16 16" version="1.1" width="16" data-view-component="true"><path fill-rule="evenodd" d="M0 6.75C0 5.784.784 5 1.75 5h1.5a.75.75 0 010 1.5h-1.5a.25.25 0 00-.25.25v7.5c0 .138.112.25.25.25h7.5a.25.25 0 00.25-.25v-1.5a.75.75 0 011.5 0v1.5A1.75 1.75 0 019.25 16h-7.5A1.75 1.75 0 010 14.25v-7.5z"></path><path fill-rule="evenodd" d="M5 1.75C5 .784 5.784 0 6.75 0h7.5C15.216 0 16 .784 16 1.75v7.5A1.75 1.75 0 0114.25 11h-7.5A1.75 1.75 0 015 9.25v-7.5zm1.75-.25a.25.25 0 00-.25.25v7.5c0 .138.112.25.25.25h7.5a.25.25 0 00.25-.25v-7.5a.25.25 0 00-.25-.25h-7.5z"></path></svg>'
const svgCheck =
  '<svg aria-hidden="true" height="16" viewBox="0 0 16 16" version="1.1" width="16" data-view-component="true"><path fill-rule="evenodd" fill="rgb(63, 185, 80)" d="M13.78 4.22a.75.75 0 010 1.06l-7.25 7.25a.75.75 0 01-1.06 0L2.22 9.28a.75.75 0 011.06-1.06L6 10.94l6.72-6.72a.75.75 0 011.06 0z"></path></svg>'

window.addEventListener("message", (event) => {
  if (event.origin !== window.location.origin) return
  if (event.data?.type !== "nuggets-widget-wheel") return
  window.scrollBy({ left: event.data.deltaX, top: event.data.deltaY, behavior: "instant" })
})

function setWidgetActive(shell: Element, active: boolean) {
  shell.classList.toggle("widget-active", active)
  const frame = shell.querySelector<HTMLIFrameElement>("iframe.widget-frame")
  if (frame) {
    frame.classList.toggle("widget-frame-inert", !active)
    frame.tabIndex = active ? 0 : -1
    frame.setAttribute("aria-hidden", String(!active))
  }
  const activate = shell.querySelector<HTMLButtonElement>(".widget-activate")
  if (activate) {
    activate.tabIndex = active ? -1 : 0
    activate.setAttribute("aria-expanded", String(active))
    activate.setAttribute("aria-hidden", String(active))
  }
}

function deactivateWidgets(except?: Element) {
  document.querySelectorAll(".widget-shell.widget-active").forEach((shell) => {
    if (shell !== except) setWidgetActive(shell, false)
  })
}

document.addEventListener("click", (event) => {
  const target = event.target as Element | null
  if (!target?.closest(".widget-shell")) deactivateWidgets()
})

document.addEventListener("keydown", (event) => {
  if (event.key === "Escape") deactivateWidgets()
})

document.addEventListener("nav", () => {
  const controller = new AbortController()
  const { signal } = controller
  window.addCleanup(() => controller.abort())
  document.querySelectorAll(".widget-shell").forEach((shell) => {
    const frame = shell.querySelector<HTMLIFrameElement>("iframe.widget-frame")
    const activate = shell.querySelector<HTMLButtonElement>(".widget-activate")
    setWidgetActive(shell, false)
    activate?.addEventListener(
      "click",
      (event) => {
        event.stopPropagation()
        deactivateWidgets(shell)
        setWidgetActive(shell, true)
        frame?.focus({ preventScroll: true })
      },
      { signal },
    )

    if (!frame) return
    let frameDocument: Document | null = null
    const escapeFrame = (event: KeyboardEvent) => {
      if (
        event.key !== "Escape" ||
        event.defaultPrevented ||
        !shell.classList.contains("widget-active")
      )
        return
      event.preventDefault()
      event.stopPropagation()
      setWidgetActive(shell, false)
      activate?.focus({ preventScroll: true })
    }
    const bindFrame = () => {
      frameDocument?.removeEventListener("keydown", escapeFrame)
      try {
        frameDocument = frame.contentDocument
      } catch {
        frameDocument = null
      }
      frameDocument?.addEventListener("keydown", escapeFrame)
    }
    frame.addEventListener("load", bindFrame, { signal })
    bindFrame()
    window.addCleanup(() => frameDocument?.removeEventListener("keydown", escapeFrame))
  })

  const els = document.getElementsByTagName("pre")
  for (let i = 0; i < els.length; i++) {
    const codeBlock = els[i].getElementsByTagName("code")[0]
    if (codeBlock) {
      const annotated = els[i].closest<HTMLElement>(".code-annotations")
      const source =
        annotated?.querySelector("pre") === els[i]
          ? annotated.dataset.codeSource!
          : codeBlock.innerText.replace(/\n\n/g, "\n")
      const button = document.createElement("button")
      button.className = "clipboard-button"
      button.type = "button"
      button.innerHTML = svgCopy
      button.ariaLabel = "Copy source"
      function onClick() {
        navigator.clipboard.writeText(source).then(
          () => {
            button.blur()
            button.innerHTML = svgCheck
            setTimeout(() => {
              button.innerHTML = svgCopy
              button.style.borderColor = ""
            }, 2000)
          },
          (error) => console.error(error),
        )
      }
      button.addEventListener("click", onClick)
      window.addCleanup(() => button.removeEventListener("click", onClick))
      els[i].prepend(button)
    }
  }
})
