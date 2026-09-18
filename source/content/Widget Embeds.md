---
title: Widget Embeds
draft: true
---

# Widget Embeds

For line plots, points, and uncertainty intervals, prefer the shared native `chart` embed documented in `Native Charts.md`. It inherits the blog theme and supports live legend values, keyboard/touch inspection, and a data table without an iframe. Keep standalone widgets for interactions beyond that chart contract.

Author widgets as standalone HTML files under `content/widgets/`, then embed them from a note with Obsidian wikilink syntax. Keep this note as the reference; do not put widget demos on the homepage.

```md
![[widgets/sm-roofline.html|A compact interactive model embedded in the post]]
```

![[widgets/sm-roofline.html|A compact interactive model embedded in the post]]

Widgets start passive: wheel/touch scrolling stays with the article, and keyboard focus goes to the activation button rather than into the iframe. Click the pane (or activate the button with the keyboard) to give the widget its own input handling. Click outside to release it. For same-origin widgets, an unhandled Escape key inside the iframe also releases it and returns focus to the activation button without moving the page. Cross-origin viewers must provide their own Escape behavior; clicking outside still releases them.

Native charts continue passing wheel scrolling to the page even when pinned, since they have no wheel-driven zoom.

Keep widget assets beside the HTML file when possible. Relative links such as `./data.json` or `./script.js` should stay in the same widget folder.
