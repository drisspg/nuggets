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

Wrap the annotated phrase and its note in a `sidenote-pair`:

```md
This sentence needs <span class="sidenote-pair"><span class="sidenote-ref" tabindex="0" aria-describedby="extra-context-note">extra context</span>.<span id="extra-context-note" class="sidenote" role="note">Keep the sidenote short and useful.</span></span>
```

The phrase has a dotted underline, like Attention Gym's documentation. Hovering or focusing the phrase highlights both it and its note; hovering the note highlights the phrase too. Use a unique note ID and match it in `aria-describedby`. Notes remain visible in the margin on desktop and inline on smaller screens.

The older numbered style is still supported:

```md
This is the sentence that needs extra context.<span class="sidenote-ref" aria-hidden="true"></span><span class="sidenote" role="note">Keep the sidenote short and useful.</span>
```

Empty references and their notes are numbered automatically in document order; phrase-based references do not consume a number.

For explanations attached to particular code lines, use [Code Annotations](./Code-Annotations): numbered comments become interactive markers without hand-written HTML.

## Writing guidelines

- Use sidenotes for optional context, asides, definitions, and implementation details.
- Keep each note to one short paragraph.
- Keep the reference and note in the same pair; put the note after the sentence it annotates.
- Avoid stacking several sidenotes in the same paragraph.
- Keep essential arguments and required instructions in the main text.
- Use regular Markdown footnotes when the note is primarily a citation.

The layout follows the margin-note pattern popularized by Edward Tufte and used by long-form sites such as Gwern. The right-margin layout activates on wide screens; tablets and phones show the same content as an inline callout.
