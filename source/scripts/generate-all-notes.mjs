#!/usr/bin/env node
import { promises as fs } from "node:fs"
import path from "node:path"
import { fileURLToPath } from "node:url"
import matter from "gray-matter"

const SCRIPT_DIR = path.dirname(fileURLToPath(import.meta.url))
const CONTENT_DIR = path.resolve(SCRIPT_DIR, "..", "content")
const OUTPUT_FILE = path.join(CONTENT_DIR, "All Notes.md")
const SKIP_SLUGS = new Set(["index", "All Notes"])

const MONTHS = [
  "January",
  "February",
  "March",
  "April",
  "May",
  "June",
  "July",
  "August",
  "September",
  "October",
  "November",
  "December",
]

function isoDate(value) {
  if (value instanceof Date) {
    return value.toISOString().slice(0, 10)
  }
  return String(value).slice(0, 10)
}

function formatDate(iso) {
  const d = new Date(`${iso}T00:00:00Z`)
  return `${MONTHS[d.getUTCMonth()]} ${d.getUTCDate()}, ${d.getUTCFullYear()}`
}

async function collectEntries() {
  const files = await fs.readdir(CONTENT_DIR)
  const entries = []

  for (const file of files) {
    if (!file.endsWith(".md")) continue
    const slug = file.replace(/\.md$/, "")
    if (SKIP_SLUGS.has(slug)) continue

    const raw = await fs.readFile(path.join(CONTENT_DIR, file), "utf8")
    const { data } = matter(raw)
    if (data.draft) continue
    if (!data.date) continue

    entries.push({ slug, iso: isoDate(data.date) })
  }

  entries.sort((a, b) => b.iso.localeCompare(a.iso))
  return entries
}

function renderMarkdown(entries) {
  const groups = new Map()
  for (const entry of entries) {
    const year = entry.iso.slice(0, 4)
    if (!groups.has(year)) groups.set(year, [])
    groups.get(year).push(entry)
  }

  const lines = ["---", "title: All Notes", "---", ""]
  for (const [year, items] of groups) {
    lines.push(`## ${year}`, "")
    for (const item of items) {
      lines.push(`- [[${item.slug}]] — ${formatDate(item.iso)}`)
    }
    lines.push("")
  }
  return lines.join("\n")
}

async function main() {
  const entries = await collectEntries()
  const next = renderMarkdown(entries)

  let prev = ""
  try {
    prev = await fs.readFile(OUTPUT_FILE, "utf8")
  } catch (err) {
    if (err.code !== "ENOENT") throw err
  }

  if (prev === next) {
    console.log(`All Notes.md already up to date (${entries.length} entries)`)
    return
  }

  await fs.writeFile(OUTPUT_FILE, next)
  console.log(`Wrote All Notes.md with ${entries.length} entries`)
}

main().catch((err) => {
  console.error(err)
  process.exit(1)
})
