---
title: Sidenotes
draft: true
---

Sidenotes put optional context next to the sentence that needs it on wide screens. On narrower screens, they become inline notes so the text remains readable without horizontal scrolling.

## Enable sidenotes

Add the `sidenotes` class to the note frontmatter:

```yaml
cssclasses:
  - sidenotes
```

Place the reference and note immediately after the relevant sentence:

```md
This is the sentence that needs extra context.<span class="sidenote-ref" aria-hidden="true"></span><span class="sidenote" role="note">Keep the sidenote short and useful.</span>
```

The reference and note are numbered automatically in document order.

## Writing guidelines

- Use sidenotes for optional context, asides, definitions, and implementation details.
- Keep each note to one short paragraph.
- Put the markup directly after the sentence it annotates.
- Avoid stacking several sidenotes in the same paragraph.
- Keep essential arguments and required instructions in the main text.
- Use regular Markdown footnotes when the note is primarily a citation.

The layout follows the margin-note pattern popularized by Edward Tufte and used by long-form sites such as Gwern. The right-margin layout activates on wide screens; tablets and phones show the same content as an inline callout.
