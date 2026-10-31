# TorchInductor Walkthrough (guest post)

Source for `content/TorchInductor Walkthrough.md` and its Colab notebook.

- `executed.ipynb`: Elias Ellison's notebook, executed with outputs. Internal image links were replaced (the scheduler graph is regenerated; the inlining diagram is still a TODO), host metadata was stripped, and the four ON/OFF demo headers print with `flush=True` so they precede their stderr logs.
- `colab_setup.ipynb`: cells prepended to the Colab copy (GPU runtime note and nightly install).

Executed on an H100 with `torch 2.16.0.dev20261031+cu132` and Triton 3.9.0 via
`jupyter nbconvert --to notebook --execute`.

Regenerate from `source/visuals/notebooks`:

```sh
python3 notebook_to_post.py inductor-walkthrough/executed.ipynb \
  --markdown /tmp/inductor_body.md \
  --colab ../../content/media/inductor-walkthrough/inductor_walkthrough.ipynb \
  --colab-setup inductor-walkthrough/colab_setup.ipynb \
  --drop-line 'dynamo_pgo force disabled' --drop-line '^  warn_once\($'
```

Paste the Markdown below the post's hand-written header, then point the scheduler-graph image at `./media/inductor-walkthrough/sum-sum-scheduler-graph.svg`.
