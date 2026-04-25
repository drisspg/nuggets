const themeStorageKey = "nuggets-theme"
const currentTheme = localStorage.getItem(themeStorageKey) ?? "dark"
document.documentElement.setAttribute("saved-theme", currentTheme)

const emitThemeChangeEvent = (theme: "light" | "dark") => {
  const event: CustomEventMap["themechange"] = new CustomEvent("themechange", {
    detail: { theme },
  })
  document.dispatchEvent(event)
}

document.addEventListener("nav", () => {
  const switchTheme = (_e: Event) => {
    const newTheme =
      document.documentElement.getAttribute("saved-theme") === "dark" ? "light" : "dark"
    document.documentElement.setAttribute("saved-theme", newTheme)
    localStorage.setItem(themeStorageKey, newTheme)
    emitThemeChangeEvent(newTheme)
  }

  const themeButton = document.querySelector("#darkmode") as HTMLButtonElement
  themeButton.addEventListener("click", switchTheme)
  window.addCleanup(() => themeButton.removeEventListener("click", switchTheme))
})
