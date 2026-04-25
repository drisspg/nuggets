---
title: Widget Embeds
draft: true
---

# Widget Embeds

Author widgets as standalone HTML files under `content/widgets/`, then embed them from a note with Obsidian wikilink syntax. Keep this note as the reference; do not put widget demos on the homepage.

```md
![[widgets/sm-roofline.html|A compact interactive model embedded in the post]]
```

![[widgets/sm-roofline.html|A compact interactive model embedded in the post]]

Keep widget assets beside the HTML file when possible. Relative links such as `./data.json` or `./script.js` should stay in the same widget folder.
