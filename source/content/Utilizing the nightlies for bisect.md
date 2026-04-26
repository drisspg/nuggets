---
title: Utilizing the nightlies for bisect
date: 2025-10-04
tags:
  - tool-tips
  - pytorch
---

#### Written: October 4, 2025

# Utilizing the nightlies for bisect

So I am trying to bisect what commit fixed this: https://github.com/pytorch/pytorch/issues/164608, to see if we can cherry-pick the changes. Instead of doing a large sweep I (mostly claude) hacked up a lil nightly wheel bisector:

https://github.com/drisspg/transformer_nuggets/blob/main/scripts/bisect_nightly.py

Nikita Shulga has told me he does this from time to time so nothing too special here. That being said it concretely helped me find the commit that fixed this: https://github.com/pytorch/pytorch/issues/164608#issuecomment-3368529834

Workflow is basically:
1. Write a python script that errors for the needle you are trying to find
2. Run this script, pass your error script as an argument, it will pull all the most recent nightlies ~ 2months, and run the script.
3. You will hopefully get 2 nightly commits, 1 working and one failing. You can then use these nightly commits to to get your git bisect indices

Sample output

```
❯ python scripts/bisect_nightly.py ~/repros/flex/cache.py
Detecting Python version in conda environment...
Python version: cp312

Fetching available nightly builds...
✓ Found 59 nightly builds available for cp312
  Date range: 20250806 to 20251004
  Binary search will require ~6 tests

Testing 59 nightly builds from 20250806 to 20251004
Using conda environment: nightly
Python version: cp312
Running test script: /home/dev/repros/flex/cache.py
Mode: Finding when bug fix occurred

  ❌ 20250904: FAILED - Bug exists (commit: 6f6ce1a7)
  ✅ 20250920: PASSED - Bug is fixed (commit: d60c6e4a)
  ✅ 20250912: PASSED - Bug is fixed (commit: 616c50a1)
  ❌ 20250907: FAILED - Bug exists (commit: d110794e)
  ❌ 20250909: FAILED - Bug exists (commit: 1ff17af8)
  ❌ 20250910: FAILED - Bug exists (commit: 75e7f49f)
Testing 20250910 (torch-2.10.0): 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 6/6 [02:03<00:00, 20.55s/test]

============================================================
Bug was FIXED between:
  Last bad:   20250910
  First good: 20250912
Commit SHA range:
                Older commit: 75e7f49f9c70116d7c4f8f86c3d0688ade306284 (20250910)
                Newer commit: 616c50a126f79be461a1977b3192ad1153275d41 (20250912)

              Note: These SHAs may be from the nightly release branch.
              Extract the actual main branch SHAs from the nightly release commits:
                cd <pytorch>
                git log --format='%H %s %b' 75e7f49f9c70116d7c4f8f86c3d0688ade306284 -1
                git log --format='%H %s %b' 616c50a126f79be461a1977b3192ad1153275d41 -1
                # Look for the main branch SHA in parentheses in the commit message

              To bisect locally (finding when bug was FIXED):
cd <pytorch>
git bisect start
git bisect old <older_main_sha>   # Older commit - bug exists (test fails)
git bisect new <newer_main_sha>   # Newer commit - bug fixed (test passes)
# Build and test each commit:
python setup.py develop && python /home/dev/repros/flex/cache.py
git bisect old   # If test FAILS (bug still exists)
git bisect new   # If test PASSES (bug is fixed)
============================================================
```

It has a mode for testing when something was broken and then fixed (uncommon) or testing for regressions.

Hope this helps someone else :)
