-------------------------

Tracker is object used to track the pose of one object
Detector has the basic cv loop and detects april tags
DataCollector used for data management

The active detector, tag IDs, tag-to-object transforms, pose math, live overlay,
and Parquet writer now live in
the standalone `Detecting.py` pipeline. `Detecting.py` is the standalone launcher
around that shared module, so standalone checks and integrated pull collection
use identical tracking logic. The saved Parquet rows are always written in the
Franka base frame; the camera-to-base calibration is applied once in the
tracking writer and then carried forward as metadata for replay/debugging.

```bash
cd at-tracking
python Detecting.py --output tracking.parquet
# Add --no-display for headless collection.
```

The window shows the live overlay in either camera space or an alternate
object-aligned overlay, depending on the `--use-reference-frame` flag. Axis
colors are X red, Y green, and Z blue. In this standalone wrapper, `q`/Escape
ends collection and writes the file; in the integrated pull test, those keys
only hide the diagnostic window.
## Unified Parquet replay

`Replay.py` browses a compiled unified system-ID Parquet over the current
RealSense color feed. It is intended for checking the camera/base-frame
calibration visually.

```bash
cd at-tracking
python Replay.py ../pull_unified.parquet
```

The unified file stores geometry in Franka base `O`. Replay reads the
run-specific `camera_to_base_4x4_used` matrix from its metadata, inverts it
(`base O -> camera`), and projects TCP, apple, apple axes, and all
Branch/Spur/Apple woody start/end chords onto the live image. The tracking
writer stores the camera calibration in metadata, and the tracking compiler
refuses non-base input instead of silently reinterpreting it.

Controls: `Space` play/pause; `Right`/`d` and `Left`/`a` step one saved row;
`Home`/`r` first row; `End` last row; `+`/`-` change speed; `q`/`Esc` quit.

The overlay only establishes whether the saved geometry is consistent with the
current physical scene. It cannot validate a file if the robot, tree, or
camera has moved since the recording.

--------------
Current tags printed tag36h11 with tags 0-8 and total size 23 / tag size (according to chaitanyantr.github.io/apriltag.html) is 18.4 mm
