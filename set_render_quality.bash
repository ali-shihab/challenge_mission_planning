#!/bin/bash
# =============================================================================
# set_render_quality.bash — trade simulator visual quality for camera throughput
#
# Recording needs the Gazebo GUI, but the GUI and the drone's camera sensor
# render through the same engine and compete for the same GPU/CPU budget on this
# VM. That competition is what starves the ArUco detector: measured 44% marker
# confirmation with the GUI up against 90-100% headless.
#
# This lowers the cost of both so detection holds up while recording.
#   low     camera 640x480, shadows off   (use if detection is poor)
#   restore original settings             (use before collecting data again)
#
# Camera intrinsics are scaled with the resolution, since fx/fy/cx/cy are
# defined against the image size and would otherwise be inconsistent.
#
# Usage:
#   ./set_render_quality.bash low
#   ./set_render_quality.bash restore
#   ./set_render_quality.bash status
# =============================================================================
set -uo pipefail

AS2_SHARE="$HOME/mission_planning_ws/install/as2_gazebo_assets/share/as2_gazebo_assets"
CAM="$AS2_SHARE/models/hd_camera/hd_camera.sdf.jinja"
WORLD="$AS2_SHARE/worlds/empty.sdf.jinja"
BAK_SUFFIX=".fullquality"

usage() { sed -n '2,22p' "$0"; exit 0; }
[[ $# -lt 1 ]] && usage

backup_once() {
  [[ -f "$1$BAK_SUFFIX" ]] || cp "$1" "$1$BAK_SUFFIX"
}

case "$1" in
  low)
    for f in "$CAM" "$WORLD"; do
      [[ -f "$f" ]] || { echo "missing: $f"; exit 1; }
      backup_once "$f"
    done

    # 1280x960 -> 640x480, with intrinsics halved to match.
    sed -i 's|<width>1280</width>|<width>640</width>|'   "$CAM"
    sed -i 's|<height>960</height>|<height>480</height>|' "$CAM"
    sed -i 's|<fx>1108.5</fx>|<fx>554.25</fx>|'           "$CAM"
    sed -i 's|<fy>1108.5</fy>|<fy>554.25</fy>|'           "$CAM"
    sed -i 's|<cx>640.5</cx>|<cx>320.5</cx>|'             "$CAM"
    sed -i 's|<cy>480.5</cy>|<cy>240.5</cy>|'             "$CAM"

    # Shadow casting is one of the most expensive parts of the scene.
    sed -i 's|<cast_shadows>true</cast_shadows>|<cast_shadows>false</cast_shadows>|' "$WORLD"

    echo "render quality: LOW"
    echo "  camera  640x480 (intrinsics scaled)"
    echo "  shadows off"
    echo "Relaunch the simulator for this to take effect."
    echo "Run './set_render_quality.bash restore' before collecting data again."
    ;;

  restore)
    n=0
    for f in "$CAM" "$WORLD"; do
      if [[ -f "$f$BAK_SUFFIX" ]]; then
        mv -f "$f$BAK_SUFFIX" "$f"; n=$((n+1))
      fi
    done
    if (( n )); then
      echo "render quality: RESTORED ($n file(s)). Relaunch the simulator."
    else
      echo "nothing to restore; already at original settings."
    fi
    ;;

  status)
    echo -n "camera resolution : "
    grep -o '<width>[0-9]*</width>' "$CAM" | head -1 | tr -d '<width>/' | xargs echo -n
    grep -o '<height>[0-9]*</height>' "$CAM" | head -1 | sed 's|[^0-9]||g' | xargs -I{} echo " x {}"
    echo -n "shadows           : "
    grep -o '<cast_shadows>[a-z]*</cast_shadows>' "$WORLD" | head -1 | sed 's|.*>\(.*\)<.*|\1|'
    echo -n "backups present   : "
    ls "$CAM$BAK_SUFFIX" "$WORLD$BAK_SUFFIX" >/dev/null 2>&1 && echo "yes (currently modified)" || echo "no (original settings)"
    ;;

  *) usage ;;
esac
