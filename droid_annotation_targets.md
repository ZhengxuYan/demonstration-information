# DROID Annotation Targets

This notes file records the DROID / pen-in-cup data locations found from the local machine and from the Tian chat context.

## Jason-Collected Pen-in-Cup Data

Local root:

```text
/Users/jasonyan/Desktop/droid-main/data
```

Trajectory counts found locally:

| Split | Count | Root |
| --- | ---: | --- |
| Success | 32 | `/Users/jasonyan/Desktop/droid-main/data/success` |
| Failure | 23 | `/Users/jasonyan/Desktop/droid-main/data/failure` |

Each trajectory directory has this shape:

```text
.../<success-or-failure>/<YYYY-MM-DD>/<trajectory_name>/
  trajectory.h5
  recordings/MP4/18650758.mp4
  recordings/MP4/25916956.mp4
  recordings/H264/18650758.mp4
  recordings/H264/25916956.mp4
  recordings/SVO/*.svo2
```

Useful local review pages:

```text
/Users/jasonyan/Desktop/droid-main/trajectory_review/index.html
/Users/jasonyan/Desktop/droid-main/success_video_review/index.html
/Users/jasonyan/Desktop/droid-main/deploy_success_video_review/index.html
```

Public page previously shared:

```text
https://deploysuccessvideoreview.vercel.app/
```

Cluster path previously sent to Tian:

```text
/scr/jasonyan/droid_success
```

## Original DROID Dataset / PennPAL Pen-in-Cup Data

Tian's chat says the DROID dataset is on ILIAD here:

```text
/iliad/group/datasets/droid
/iliad/group/datasets/droid_raw
```

The local kNN test manifest points to original DROID / R2D2 PennPAL data on Google Cloud Storage:

```text
/Users/jasonyan/Desktop/droid_knn_results/droid_pen_in_cup_knn_test/pen_in_cup_manifest.csv
```

The manifest currently contains one unique original episode id:

```text
gs://xembodiment_data/r2d2/r2d2-data-full/PennPAL/success/2023-04-29/Sat_Apr_29_22:16:23_2023/recordings/MP4--gs://xembodiment_data/r2d2/r2d2-data-full/PennPAL/success/2023-04-29/Sat_Apr_29_22:16:23_2023/trajectory.h5
```

The frames exported for that kNN page are in:

```text
/Users/jasonyan/Desktop/droid_knn_results/droid_pen_in_cup_knn_test/images
```

## Access Notes

- From this local machine, `/iliad`, `/scr`, and `/iris` are not mounted.
- `ssh iliad-hgx-1` failed because the hostname could not be resolved from the current network environment.
- To enumerate the full original DROID pen-in-cup set, run the search from a shell that can access ILIAD, then look under `/iliad/group/datasets/droid` and `/iliad/group/datasets/droid_raw` for `pen`, `cup`, or `PennPAL`.
