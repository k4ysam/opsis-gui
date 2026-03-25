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
import time

import numpy as np
import SimpleITK as sitk


class ThresholdSegmenter:
    """Performs threshold-based and seed-based segmentation."""

    @staticmethod
    def _log(message: str):
        print(f"[timing] {message}", flush=True)

    # ------------------------------------------------------------------
    # Skin segmentation
    # ------------------------------------------------------------------

    def segment_skin(
        self,
        image: sitk.Image,
        lower_hu: float = -200.0,
        upper_hu: float = 500.0,
        closing_radius: int = 3,
        shrink_factor: int = 2,
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
        shrink_factor : int
            Optional isotropic shrink factor applied before segmentation to
            reduce computation time for large CT volumes.

        Returns
        -------
        sitk.Image
            Binary uint8 label map (1 = skin/body, 0 = background).
        """
        t0 = time.perf_counter()
        work_image = image
        if shrink_factor > 1:
            work_image = sitk.Shrink(image, [shrink_factor] * 3)
            self._log(
                "segment_skin: "
                f"downsampled by {shrink_factor} to size={work_image.GetSize()}"
            )

        # 1. Threshold
        binary = sitk.BinaryThreshold(
            work_image,
            lowerThreshold=float(lower_hu),
            upperThreshold=float(upper_hu),
            insideValue=1,
            outsideValue=0,
        )

        # 2. Morphological closing to fill gaps (cast to uint8 for filter)
        binary = sitk.Cast(binary, sitk.sitkUInt8)
        kernel = sitk.sitkBall
        effective_radius = max(1, int(round(closing_radius / max(shrink_factor, 1))))
        binary = sitk.BinaryMorphologicalClosing(binary, [effective_radius] * 3, kernel)

        # 3. Largest connected component
        binary = self._keep_largest_component(binary)

        self._log(
            "segment_skin: "
            f"size={work_image.GetSize()} closing_radius={effective_radius} "
            f"took {time.perf_counter() - t0:.2f}s"
        )
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
        roi_radius: int = 48,
        seed_margin: float = 45.0,
        smooth_sigma: float = 1.5,
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
        t0 = time.perf_counter()
        size = image.GetSize()
        seed_x, seed_y, seed_z = (int(seed_index[0]), int(seed_index[1]), int(seed_index[2]))

        if roi_radius > 0:
            roi_index = [
                max(0, seed_x - roi_radius),
                max(0, seed_y - roi_radius),
                max(0, seed_z - roi_radius),
            ]
            roi_upper = [
                min(size[0], seed_x + roi_radius + 1),
                min(size[1], seed_y + roi_radius + 1),
                min(size[2], seed_z + roi_radius + 1),
            ]
            roi_size = [max(1, hi - lo) for lo, hi in zip(roi_index, roi_upper)]
            roi_seed = (
                seed_x - roi_index[0],
                seed_y - roi_index[1],
                seed_z - roi_index[2],
            )
            work_image = sitk.RegionOfInterest(image, roi_size, roi_index)
        else:
            roi_index = [0, 0, 0]
            roi_size = list(size)
            roi_seed = (seed_x, seed_y, seed_z)
            work_image = image

        smoothed_image = sitk.DiscreteGaussian(work_image, variance=float(smooth_sigma) ** 2)
        seed_value = float(smoothed_image.GetPixel(*roi_seed))
        adaptive_lower = max(float(lower_hu), seed_value - float(seed_margin))
        adaptive_upper = min(float(upper_hu), seed_value + float(seed_margin))
        if adaptive_upper <= adaptive_lower:
            adaptive_lower = min(float(lower_hu), seed_value - 10.0)
            adaptive_upper = max(float(upper_hu), seed_value + 10.0)

        threshold_mask = sitk.BinaryThreshold(
            smoothed_image,
            lowerThreshold=adaptive_lower,
            upperThreshold=adaptive_upper,
            insideValue=1,
            outsideValue=0,
        )
        binary = sitk.ConfidenceConnected(
            smoothed_image,
            seedList=[roi_seed],
            numberOfIterations=2,
            multiplier=1.8,
            initialNeighborhoodRadius=1,
            replaceValue=replace_value,
        )
        binary = sitk.And(
            sitk.Cast(binary, sitk.sitkUInt8),
            sitk.Cast(threshold_mask, sitk.sitkUInt8),
        )
        binary = self._keep_largest_component(binary)
        binary = sitk.BinaryMorphologicalClosing(binary, [1, 1, 1], sitk.sitkBall)
        if roi_radius > 0:
            full_binary = sitk.Image(size, sitk.sitkUInt8)
            full_binary.CopyInformation(image)
            binary = sitk.Paste(full_binary, sitk.Cast(binary, sitk.sitkUInt8), roi_size, (0, 0, 0), roi_index)
        self._log(
            "segment_target: "
            f"seed={seed_value:.1f} bounds=({adaptive_lower:.1f}, {adaptive_upper:.1f}) "
            f"took {time.perf_counter() - t0:.2f}s"
        )
        return sitk.Cast(binary, sitk.sitkUInt8)

    def segment_from_seeds(
        self,
        image: sitk.Image,
        inside_seed_mask: np.ndarray,
        outside_seed_mask: np.ndarray,
        smooth_sigma: float = 1.0,
    ) -> sitk.Image:
        """Segment a target from inside/outside seed masks.

        This follows the SlicerOpenNav interaction model more closely:
        the user paints foreground and background hints, then the target
        is interpolated between them.
        """
        t0 = time.perf_counter()
        inside_count = int(np.count_nonzero(inside_seed_mask))
        outside_count = int(np.count_nonzero(outside_seed_mask))

        result = sitk.Image(image.GetSize(), sitk.sitkUInt8)
        result.CopyInformation(image)
        if inside_count == 0:
            return result

        inside = sitk.GetImageFromArray(inside_seed_mask.astype(np.uint8, copy=False))
        inside.CopyInformation(image)
        outside = sitk.GetImageFromArray(outside_seed_mask.astype(np.uint8, copy=False))
        outside.CopyInformation(image)

        inside = sitk.BinaryDilate(inside, [1, 1, 1], sitk.sitkBall)
        if outside_count > 0:
            outside = sitk.BinaryDilate(outside, [1, 1, 1], sitk.sitkBall)

        smoothed = sitk.DiscreteGaussian(image, variance=float(smooth_sigma) ** 2)
        gradient = sitk.GradientMagnitudeRecursiveGaussian(
            smoothed,
            sigma=max(0.5, float(smooth_sigma)),
        )

        markers = sitk.Image(image.GetSize(), sitk.sitkUInt32)
        markers.CopyInformation(image)
        markers = sitk.Cast(inside, sitk.sitkUInt32)

        if outside_count > 0:
            markers = markers + 2 * sitk.Cast(outside, sitk.sitkUInt32)
        else:
            # If the user did not paint an outside region yet, use the border
            # as a weak background prior so the interpolation still has a stop.
            border = sitk.Image(image.GetSize(), sitk.sitkUInt8)
            border.CopyInformation(image)
            size = image.GetSize()
            for z in (0, size[2] - 1):
                for y in range(size[1]):
                    for x in range(size[0]):
                        border.SetPixel(x, y, z, 1)
            for z in range(size[2]):
                for y in (0, size[1] - 1):
                    for x in range(size[0]):
                        border.SetPixel(x, y, z, 1)
            for z in range(size[2]):
                for y in range(size[1]):
                    for x in (0, size[0] - 1):
                        border.SetPixel(x, y, z, 1)
            markers = markers + 2 * sitk.Cast(border, sitk.sitkUInt32)

        watershed = sitk.MorphologicalWatershedFromMarkers(
            gradient,
            markers,
            markWatershedLine=False,
            fullyConnected=False,
        )
        binary = sitk.Equal(watershed, 1)
        binary = sitk.BinaryMorphologicalClosing(
            sitk.Cast(binary, sitk.sitkUInt8),
            [1, 1, 1],
            sitk.sitkBall,
        )
        binary = self._keep_largest_component(binary)
        self._log(
            "segment_from_seeds: "
            f"inside={inside_count:,} outside={outside_count:,} "
            f"took {time.perf_counter() - t0:.2f}s"
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
