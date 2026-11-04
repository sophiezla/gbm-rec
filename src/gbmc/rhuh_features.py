"""Features f1-f5 and the C0T distance on RHUH-GBM (plan Step 4.1).

Same ``gbmc.features.build`` as LUMIERE, on each patient's early post-op grid.
Only the barriers need adapting:

* falx: gbm-4d derives it at the MNI midline (x = 0) through LUMIERE's shipped
  atlas transform. RHUH-GBM images are already in SRI24, so the "transform to
  the atlas" is a translation by the SRI24 midline position,
  ``config.yaml:rhuh.sri24_midline_x_mm``. The midline value and the sequence
  used for the white-matter test (``rhuh.falx_t1_sequence``) are decisions to
  record before scoring (plan Step 4.1), so both are null until then and
  reading them raises.
* ventricles: gbm-4d's segmentation on the post-op T1ce, lesion excluded, as on LUMIERE.
"""

from __future__ import annotations

import numpy as np
import SimpleITK as sitk

from gbmc import config, features


def falx(reference: sitk.Image, brain: np.ndarray, t1_z: np.ndarray) -> np.ndarray:
    from gbm4d import barriers

    midline = float(config.get("rhuh.sri24_midline_x_mm"))
    to_midline = sitk.TranslationTransform(3, (-midline, 0.0, 0.0))
    return barriers.falx_barrier(reference, to_midline, brain, t1_z)


def build(reference: sitk.Image, brain: np.ndarray, target: np.ndarray,
          flair_abnormal: np.ndarray, t1_z: np.ndarray, lesion: np.ndarray
          ) -> tuple[features.FeatureVolumes, dict]:
    from gbm4d import ventricles

    brain_img = sitk.GetImageFromArray(brain.astype(np.uint8))
    brain_img.CopyInformation(reference)
    lesion_img = sitk.GetImageFromArray(lesion.astype(np.uint8))
    lesion_img.CopyInformation(reference)
    vent_img, vent_qc = ventricles.segment_ventricles(reference, brain_img, lesion_img)
    vent = sitk.GetArrayFromImage(vent_img) > 0 if vent_qc.ok else np.zeros(brain.shape, bool)
    falx_mask = falx(reference, brain, t1_z)
    vols = features.build(reference, brain, target, flair_abnormal, falx_mask, vent,
                          float(config.get("tournament.ventricle_depth_mm")))
    return vols, {"ventricles": vent_qc.as_dict() | {"used": bool(vent.any())},
                  "falx_voxels": int(falx_mask.sum())}
