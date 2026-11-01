# TorchInductor Walkthrough (guest post)

Source for `content/TorchInductor Walkthrough.md` and its Colab notebook.

- `executed.ipynb`: Elias Ellison's notebook, executed with outputs. Internal image links now point at copies in content/media (the scheduler graph is regenerated; the inlining diagram is the original), the "Other PR Topics Not Covered In Depth" section was dropped, and kernel metadata and host paths in outputs were stripped. Cell sources are Elias's verbatim, but the outputs come from a run where the four ON/OFF demo headers printed with `flush=True`; without it Jupyter shows each `=== … ===` header after the stderr logs it labels.
- `colab_setup.ipynb`: cells prepended to the Colab copy (GPU runtime note and nightly install).

Executed on an H100 with `torch 2.16.0.dev20261031+cu132` and Triton 3.9.0 via
`jupyter nbconvert --to notebook --execute`.

Regenerate from `source/visuals/notebooks`:

```sh
python3 notebook_to_post.py inductor-walkthrough/executed.ipynb \
  --markdown /tmp/inductor_body.md \
  --colab ../../content/media/inductor-walkthrough/inductor_walkthrough.ipynb \
  --colab-setup inductor-walkthrough/colab_setup.ipynb \
  --output-language python \
  --drop-line 'dynamo_pgo force disabled' --drop-line '^  warn_once\($'
```

Do not run Prettier on the post (it is in `.prettierignore`), so Elias's Markdown stays byte-for-byte. Paste the Markdown below the post's hand-written header, then point the scheduler-graph image at `./media/inductor-walkthrough/sum-sum-scheduler-graph.svg`.
