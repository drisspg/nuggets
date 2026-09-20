---
title: Code Annotations
draft: true
---

Add `annotate` to a code fence, mark lines with numbered comments, then write the explanations as an ordered Markdown list immediately below it. Hover over a marker to preview its note; click or tap to keep it open. The associated lines light up together.

## Copy this pattern

````md
```python annotate title="Reference choice" showLineNumbers
reference = gates[midpoint]  # (1)!
q_scaled = q * torch.exp2(gates - reference)  # (2)!
k_scaled = k * torch.exp2(reference - gates)  # (2)!
```

1. The reference can come from a later token. In real arithmetic, the common reference cancels.
2. Both operands are rescaled. Materializing them separately in finite precision can make the reference choice matter.
````

## Rendered example

```python annotate title="Reference choice" showLineNumbers {2}
reference = gates[midpoint]  # (1)!
q_scaled = q * torch.exp2(gates - reference)  # (2)!
k_scaled = k * torch.exp2(reference - gates)  # (2)!
```

1. The reference can come from a **later token**. In real arithmetic, the common reference cancels.
2. Both operands are rescaled: $2^{g-r}$ and $2^{r-g}$.

   Materializing them separately in finite precision can make the reference choice matter. See [KDA Doesn't Care About the Future](./KDA-Future-Token-Leakage).

## Different comments and languages

Use the comment syntax appropriate for the language:

| Language style              | Marker comment  |
| --------------------------- | --------------- |
| Python, shell               | `# (1)!`        |
| JavaScript, TypeScript, C++ | `// (1)!`       |
| SQL, Lua                    | `-- (1)!`       |
| C-style block comment       | `/* (1)! */`    |
| HTML comment                | `<!-- (1)! -->` |

Put the marker comment at the end of a line. Its contents should be only markers, not additional prose. Other ordinary code comments remain unchanged. Within an opted-in block, these suffixes are reserved annotation syntax, not a programming-language-aware comment parser.

A line can have several explanations:

```javascript annotate
const result = await evaluate(input) // (1)! (2)!
```

1. Evaluation completes before the next statement runs.
2. `result` holds the resolved value, not the promise.

Reuse a number on several lines when one explanation covers them all. Numbers restart at **1 for each code block**. Definitions are numbered by their position in the following list; ordinary Markdown lists written with repeated `1.` prefixes work too.

Notes support normal Markdown: emphasis, links, inline code, math, paragraphs, and nested lists. Keep them short enough to read beside the code. Code-block titles, line numbers, and highlighted lines such as `{2-3}` keep working.

## Reader behavior

- Hover over a marker to preview its note without moving keyboard focus. Move into the note to keep reading; leaving both dismisses the preview. Escape also dismisses it without moving focus.
- Click or tap to keep a note open, or tab to a marker and press Enter or Space. Escape or Close dismisses it and returns focus; clicking the same marker or outside also dismisses it.
- Notes use native browser popovers so they aren't clipped by horizontally scrolling code blocks. Long notes scroll within the viewport.
- The code copy button copies the **clean code**, without annotation comments or badge numbers.
- With JavaScript disabled, or without browser popover support, notes remain visible below the code and the numbered markers link to them.
- A fence without `annotate` is unchanged, even if its code happens to contain `(1)!`.

An opted-in block needs at least one marker and an immediately following ordered list starting at 1. Every referenced number needs a note, and every note needs a reference. A mismatch fails the build with the note's file path rather than silently pairing the wrong explanation. Use annotations on ordinary code fences, not `chart`, `plotly`, or `mermaid` embeds.

## Validation

From `source/`, run `npm run test:annotations` for the Markdown/highlighter transformation checks. The browser regression checks rendering, copy, keyboard/touch interaction, themes, navigation, and the no-JavaScript fallback.

As with native chart checks, Playwright is an ephemeral QA dependency, not a site dependency:

```sh
pw_dir=$(mktemp -d) && npm install --prefix "$pw_dir" --no-save playwright@1.57.0 && "$pw_dir/node_modules/.bin/playwright" install chromium && NODE_PATH="$pw_dir/node_modules" npm run test:annotations:browser
```
