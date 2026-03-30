"""ThresholdSegmenter: SimpleITK-based medical image segmentation.

Two modes:
  - ``segment_skin``: global HU threshold + morphological closing + largest
    connected component.  Replaces Slicer's SegmentEditor threshold tool.
  - ``segment_target``: connected-threshold seeded at a user-clicked voxel.
    Replaces Slicer's "paint + seed" workflow.

All operations run on SimpleITK images (LPS, float32).  Results are returned
as uint8 binary label maps (1 = foreground, 0 = background).

Usage::

    seg = ThresholdSegmenter()
    skin_label = seg.segment_skin(sitk_image, lower_hu=-200, upper_hu=500)
    target_label = seg.segment_target(sitk_image, seed_index=(ix, iy, iz),
                                      lower_hu=50, upper_hu=300)
"""

from __future__ import annotations

from typing import Tuple

import SimpleITK as sitk


class ThresholdSegmenter:
    """Performs threshold-based and seed-based segmentation."""

    # ------------------------------------------------------------------
    # Skin segmentation
    # ------------------------------------------------------------------

    def segment_skin(
        self,
        image: sitk.Image,
        lower_hu: float = -200.0,
        upper_hu: float = 500.0,
        closing_radius: int = 3,
    ) -> sitk.Image:
        """Segment the skin/body surface from a CT volume.

        Steps:
          1. Binary threshold at [lower_hu, upper_hu]
          2. Morphological closing to fill small holes
          3. Keep only the largest connected component

        Parameters
        ----------
        image : sitk.Image
            Float32 CT volume (LPS).
        lower_hu, upper_hu : float
            HU range to include (default −200 … 500 captures soft tissue + bone).
        closing_radius : int
            Radius in voxels for the closing operation.

        Returns
        -------
        sitk.Image
            Binary uint8 label map (1 = skin/body, 0 = background).
        """
        # 1. Threshold
        binary = sitk.BinaryThreshold(
            image,
            lowerThreshold=float(lower_hu),
            upperThreshold=float(upper_hu),
            insideValue=1,
            outsideValue=0,
        )

        # 2. Morphological closing to fill gaps (cast to uint8 for filter)
        binary = sitk.Cast(binary, sitk.sitkUInt8)
        kernel = sitk.sitkBall
        binary = sitk.BinaryMorphologicalClosing(binary, [closing_radius] * 3, kernel)

        # 3. Largest connected component
        binary = self._keep_largest_component(binary)

        return sitk.Cast(binary, sitk.sitkUInt8)

    # ------------------------------------------------------------------
    # Target segmentation
    # ------------------------------------------------------------------

    def segment_target(
        self,
        image: sitk.Image,
        seed_index: Tuple[int, int, int],
        lower_hu: float = 50.0,
        upper_hu: float = 300.0,
        replace_value: int = 1,
    ) -> sitk.Image:
        """Segment a target region by connected threshold seeded at one voxel.

        Parameters
        ----------
        image : sitk.Image
            Float32 CT volume (LPS).
        seed_index : tuple of int
            (ix, iy, iz) voxel index of the seed point (clicked by user).
        lower_hu, upper_hu : float
            HU range for the flood fill.
        replace_value : int
            Label value for foreground (default 1).

        Returns
        -------
        sitk.Image
            Binary uint8 label map.
        """
        binary = sitk.ConnectedThreshold(
            image,
            seedList=[seed_index],
            lower=float(lower_hu),
            upper=float(upper_hu),
            replaceValue=replace_value,
        )
        return sitk.Cast(binary, sitk.sitkUInt8)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _keep_largest_component(binary: sitk.Image) -> sitk.Image:
        """Return a label map containing only the largest connected component."""
        labeled = sitk.ConnectedComponent(binary)
        # RelabelComponent sorts by size descending; label 1 = largest
        relabeled = sitk.RelabelComponent(labeled, sortByObjectSize=True)
        # Keep only label 1
        return sitk.BinaryThreshold(
            relabeled,
            lowerThreshold=1,
            upperThreshold=1,
            insideValue=1,
            outsideValue=0,
        )
