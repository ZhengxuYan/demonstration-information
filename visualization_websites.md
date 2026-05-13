# Visualization Websites

This file indexes the experiment visualization pages shared with Tian. Links are grouped by experiment stage and marked as current when they look like the latest page for that result family.

## Current / Most Useful

| Topic | Website | What It Shows | Notes |
| --- | --- | --- | --- |
| Square PH wrist DemInf review | https://squarephwristreviewdeploy.vercel.app/ | Square PH wrist-only DemInf trajectories, score traces, and full/partial observability annotations | Current Square PH DemInf annotation page |
| Square PH BC policy NLL review | https://squarephbcpolicynllreview.vercel.app/ | BC policy per-transition `-log pi(a_i|s_i)` and trajectory-level policy scores | Current Square PH BC NLL page |
| Square PH policy-view DP report | https://square-ph-policy-view-dp-report.vercel.app/ | DP policy results for Square PH with different third-person camera views | Current DP policy-view report |
| Square PH policy-view BC report | https://square-ph-policy-view-bc-report.vercel.app/ | BC policy results for Square PH camera-view experiments; later reused for smooth discrete results | Current BC policy-view report |
| Expert200 random-post DP report | https://expert200-policy-view-dp-report.vercel.app/ | DP policy results on the Expert200 random-post-position dataset | Current Expert200 DP report |
| Combined BC and SA-VAE kNN review | https://combinedbcandsavaelatentreview.vercel.app/ | kNN visualization on newly collected random-post data using policy state latents and state-action VAE latents, with entropy / NLL context | Current kNN entropy review |
| Expert200 Qwen3-VL kNN review | https://expert200randompostqwen3vlreview.vercel.app/ | Qwen3-VL embedding nearest-neighbor visualization on Expert200 random-post data | Current Qwen3-VL kNN page |
| Successful DROID trajectory videos | https://deploysuccessvideoreview.vercel.app/ | Videos of 32 successful real-robot pen-in-cup trajectories | Current real-robot data review |
| Square PH custom third-person camera preview | https://squarephcustomviewpreview10demos.vercel.app/ | Original `agentview` plus 8 candidate third-person camera views across 10 demos | Used to select `left_close_low` |

## DemInf / MI Score Reviews

| Topic | Website | What It Shows | Notes |
| --- | --- | --- | --- |
| Initial Square DemInf score review | https://skill-deploy-38ga2cyxv9-codex-agent-deploys.vercel.app/ | Trajectory-level MI scores and videos for manually labeled label-3 Square demos | Superseded by pages with step traces |
| Square DemInf synchronized score trace | https://skill-deploy-ybounyzrsf-codex-agent-deploys.vercel.app/ | Per-demo score trace synchronized with the video | Superseded by y-axis version |
| Square DemInf score trace with y-axis | https://skill-deploy-s8zqi97eqd-codex-agent-deploys.vercel.app/ | Same as above, with y-axis added to the score curve plot | Useful historical page |
| New fixed-pose demos, wrist score curves | https://skill-deploy-zf4hivbro4-codex-agent-deploys.vercel.app/ | Forward-grab / backward-grab demos scored using previous VAE, with smoothed score curves | Before joint KSG scoring |
| Joint KSG over RoboMimic Square MH + new demos | https://skill-deploy-t23v7lllam-codex-agent-deploys.vercel.app/ | Shared-normalization KSG scores for original MH and newly collected demos | Shows separation between forward and backward grasp |
| Joint KSG with adjustable smoothing | https://skill-deploy-abbchxs2il-codex-agent-deploys.vercel.app/ | Same joint KSG result with manual smoothing-window control | Prefer this over fixed smoothing page |
| Wrist-only / agent-only DemInf views | https://skill-deploy-8rmdchp6ht-codex-agent-deploys.vercel.app/ | Separate wrist-view and third-person-view DemInf scores on combined MH + 51 random/manual demos | Pre-fused multi-view setting |
| Multi-view DemInf experiment results | https://skill-deploy-n2at00oq2s-codex-agent-deploys.vercel.app/ | Follow-up DemInf experiment page after adding fused multi-view scoring | Exact variant should be checked before reuse |
| Multi-view DemInf experiment results | https://skill-deploy-qly50rneoy-codex-agent-deploys.vercel.app/ | Follow-up DemInf experiment page paired with the previous link | Exact variant should be checked before reuse |
| Clearer multi-view rollout / score curves | https://skill-deploy-787pgjxnv7-codex-agent-deploys.vercel.app/ | Clearer videos with wrist-only, agent-only, and both-view score curves | Tian noted curves are very similar |
| DemInf score gaps | https://skill-deploy-42su8w3qza-codex-agent-deploys.vercel.app/ | Gaps between MI scores under different VAEs / camera settings | Used to inspect camera-view score differences |
| Label 1 / 2 / 3 suboptimal examples | https://skill-deploy-6yg2h9xy26-codex-agent-deploys.vercel.app/ | Selected trajectories from MH labels 1.0, 2.0, and 3.0 | Used to compare MI scores on suboptimal data |

## kNN / Latent-Space Reviews

| Topic | Website | What It Shows | Notes |
| --- | --- | --- | --- |
| Forward-grab wrist latent kNN | https://skill-deploy-9ka25c72oq-codex-agent-deploys.vercel.app/ | Nearest neighbors in learned wrist-observation latent space on forward-grab demos | Same-demo frames excluded, max one neighbor per other demo |
| Square MH wrist latent kNN | https://skill-deploy-uuh87pejzs-codex-agent-deploys.vercel.app/ | kNN on the broader Square MH dataset | Revealed more matching errors and background sensitivity |

## Policy / View Reports

| Topic | Website | What It Shows | Notes |
| --- | --- | --- | --- |
| Square PH policy-view DP report | https://square-ph-policy-view-dp-report.vercel.app/ | DP policy evaluation under camera-view variants | Current report |
| Square PH policy-view BC report | https://square-ph-policy-view-bc-report.vercel.app/ | GMM / discrete / smooth-discrete BC policy results under camera-view variants | Current report |
| Expert200 random-post DP report | https://expert200-policy-view-dp-report.vercel.app/ | DP policy evaluation on Expert200 random-post dataset | Current report |
| Square PH BC policy NLL review | https://squarephbcpolicynllreview.vercel.app/ | Per-transition NLL policy visualization | Current NLL page |

## External References / Non-Review Pages

| Topic | Website | Why It Was Shared |
| --- | --- | --- |
| Diffusion Policy | https://diffusion-policy.cs.columbia.edu/ | Related work / baseline reference |
| DROID dataset | https://droid-dataset.github.io/ | DROID dataset reference |
| DROID teleoperation docs | https://droid-dataset.github.io/droid/example-workflows/teleoperation.html | Real-robot teleoperation setup |
| DROID data collection docs | https://droid-dataset.github.io/droid/example-workflows/data-collection.html | Real-robot data collection setup |
| DROID host / Oculus setup | https://droid-dataset.github.io/droid/software-setup/host-installation.html | Oculus Quest setup |
| OpenPI | https://github.com/Physical-Intelligence/openpi | Policy code reference |
| Tian OpenPI branch | https://github.com/skybhh19/openpi/tree/pomdp_vla | Project-specific OpenPI branch |

## Notes

- Prefer the named Vercel domains over old `skill-deploy-*` links when both exist for the same result family.
- Old `skill-deploy-*` links are kept because they document intermediate decisions and may still be useful for comparisons.
- The `left_close_low` camera view selected from the preview page used:
  - `camera pos = [0.42205740, -0.23999999, 1.15230719]`
  - `camera quat = [0.81392215, 0.36066498, 0.18452251, 0.41641680]`
