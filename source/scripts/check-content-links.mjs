#!/usr/bin/env node
import { execFileSync } from "node:child_process"
import { promises as fs } from "node:fs"
import path from "node:path"
import { fileURLToPath } from "node:url"

const SCRIPT_DIR = path.dirname(fileURLToPath(import.meta.url))
const ROOT_DIR = path.resolve(SCRIPT_DIR, "..", "..")
const CONTENT_DIR = path.resolve(SCRIPT_DIR, "..", "content")

function stagedMarkdownFiles() {
  const output = execFileSync("git", ["diff", "--cached", "--name-only", "--diff-filter=ACMR"], {
    cwd: ROOT_DIR,
    encoding: "utf8",
  })

  return output
    .split("\n")
    .filter((file) => file.startsWith("source/content/") && file.endsWith(".md"))
    .map((file) => path.resolve(ROOT_DIR, file))
}

async function allMarkdownFiles(dir = CONTENT_DIR) {
  const entries = await fs.readdir(dir, { withFileTypes: true })
  const files = []

  for (const entry of entries) {
    const fullPath = path.join(dir, entry.name)
    if (entry.isDirectory()) {
      files.push(...(await allMarkdownFiles(fullPath)))
    } else if (entry.isFile() && entry.name.endsWith(".md")) {
      files.push(fullPath)
    }
  }

  return files
}

function linkTargets(markdown) {
  const targets = []
  const wikilinkPattern = /!??\[\[([^\]]+)\]\]/g

  for (const match of markdown.matchAll(wikilinkPattern)) {
    const rawTarget = match[1].split("|")[0].split("#")[0].trim()
    if (rawTarget && !rawTarget.match(/^[a-z]+:/i)) {
      targets.push(rawTarget)
    }
  }

  return targets
}

async function exists(file) {
  try {
    await fs.access(file)
    return true
  } catch {
    return false
  }
}

async function resolves(target) {
  const direct = path.resolve(CONTENT_DIR, target)
  if (await exists(direct)) return true

  if (!path.extname(target) && (await exists(`${direct}.md`))) return true

  return false
}

async function main() {
  const files = process.argv.includes("--all") ? await allMarkdownFiles() : stagedMarkdownFiles()
  const failures = []

  for (const file of files) {
    const markdown = await fs.readFile(file, "utf8")
    for (const target of linkTargets(markdown)) {
      if (!(await resolves(target))) {
        failures.push(`${path.relative(ROOT_DIR, file)} -> [[${target}]]`)
      }
    }
  }

  if (failures.length) {
    console.error("Broken content wikilinks:")
    for (const failure of failures) console.error(`  ${failure}`)
    process.exit(1)
  }

  console.log(`Checked ${files.length} staged content markdown file(s)`)
}

main().catch((err) => {
  console.error(err)
  process.exit(1)
})
