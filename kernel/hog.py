"""Provide a class to compute Histograms of Oriented Gradients
of images.
"""
from typing import Tuple

import numpy as np
from joblib import Parallel, delayed

from kernel.base import Base


def _normalize_block(block: np.ndarray, method: str, eps: float = 1e-5
                     ) -> np.ndarray:
    """Normalize a block from an image.

    Parameters
    ----------
    block : np.ndarray
        Block to normalize.
    method : str
        Method to follow, must be in {'L1', 'L2', 'L2-Hys'}
    eps : float, optional
        Constant to add to the denominator, by default 1e-5

    Returns
    -------
    np.ndarray
        Normalized block.

    Raises
    ------
    ValueError
        If normalization method given is invalid.
    """
    if method == 'L1':
        out = block / (np.sum(np.abs(block)) + eps)
    elif method == 'L1-sqrt':
        out = np.sqrt(block / (np.sum(np.abs(block)) + eps))
    elif method == 'L2':
        out = block / np.sqrt(np.sum(block ** 2) + eps ** 2)
    elif method == 'L2-Hys':
        out = block / np.sqrt(np.sum(block ** 2) + eps ** 2)
        out = np.minimum(out, 0.2)
        out = out / np.sqrt(np.sum(out ** 2) + eps ** 2)
    else:
        raise ValueError('Selected block normalization method is invalid.')
    return out


def _gradient(image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Compute the gradient along the rows and columns from an image.
    The image is expected to have multiple channels.
    In each pixel, we keep the gradient in the channel with the maximum
    intensity.

    Parameters
    ----------
    image : np.ndarray
        Image to compute the gradient from, of shape (W, H, C).

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Gradient along the rows and gradient along the columns, each of
        shape (W, H).
    """
    num_rows, num_cols, num_channels = image.shape
    gradient_row = np.empty_like(image, dtype=np.float64)
    gradient_col = np.empty_like(image, dtype=np.float64)
    gradient_norm = np.empty_like(image, dtype=np.float64)

    for i in range(num_channels):
        g_row = np.empty((num_rows, num_cols), dtype=np.float64)
        g_row[0, :] = 0
        g_row[-1, :] = 0
        g_row[1:-1, :] = image[2:, :, i] - image[:-2, :, i]
        g_col = np.empty((num_rows, num_cols), dtype=np.float64)
        g_col[:, 0] = 0
        g_col[:, -1] = 0
        g_col[:, 1:-1] = image[:, 2:, i] - image[:, :-2, i]
        gradient_row[:, :, i], gradient_col[:, :, i] = g_row, g_col

    gradient_norm = np.hypot(gradient_row, gradient_col)
    idx = gradient_norm.argmax(axis=2)
    xx, yy = np.meshgrid(np.arange(num_rows), np.arange(num_cols),
                         indexing='ij', sparse=True)
    return gradient_row[xx, yy, idx], gradient_col[xx, yy, idx]


def _histograms(norm: np.ndarray, orientation: np.ndarray,
                num_orientations: int, pixels_per_cell: Tuple[int, int]
                ) -> np.ndarray:
    """Compute the histogram of gradient orientations given
    the norm of the gradient and its orientation in all pixels.

    Parameters
    ----------
    norm : np.ndarray
        Norm of the gradient.
    orientation : np.ndarray
        Orientation of the gradient.
    num_orientations : int
        Number of orientation bins.
    pixels_per_cell : Tuple[int, int]
        Size in pixel of a cell.

    Returns
    -------
    np.ndarray
        Histogram of gradients orientations.
    """
    num_rows, num_cols = norm.shape
    cell_rows, cell_columns = pixels_per_cell
    num_cells_row = int(num_rows // cell_rows)
    num_cells_col = int(num_cols // cell_columns)
    orientation_histogram = np.zeros(
        (num_cells_row, num_cells_col, num_orientations), dtype=np.float64)

    # Useful constants
    range_rows_stop = (cell_rows + 1) // 2
    range_rows_start = -(cell_rows // 2)
    range_columns_stop = (cell_columns + 1) // 2
    range_columns_start = -(cell_columns // 2)
    number_of_orientations_per_180 = 180. / num_orientations

    for i in range(num_orientations):  # Iterate on each orientation
        # Limits of the bin
        orientation_start = number_of_orientations_per_180 * (i + 1)
        orientation_end = number_of_orientations_per_180 * i
        c = cell_columns // 2
        r = cell_rows // 2
        r_i = 0
        c_i = 0
        # Iterate on the on each cell
        while r < cell_rows * num_cells_row:
            c_i = 0
            c = cell_columns // 2
            while c < cell_columns * num_cells_col:
                total = 0
                # Iterate inside a cell
                for cell_row in range(range_rows_start, range_rows_stop):
                    cell_row_idx = r + cell_row
                    if cell_row_idx < 0 or cell_row_idx >= num_rows:
                        continue  # Row out of bounds
                    for cell_column in range(range_columns_start,
                                             range_columns_stop):
                        cell_column_idx = c + cell_column
                        if (cell_column_idx < 0 or cell_column_idx >= num_cols
                                or orientation[cell_row_idx, cell_column_idx
                                               ] >= orientation_start
                                or orientation[cell_row_idx, cell_column_idx
                                               ] < orientation_end
                            ):
                            # Column out of bounds or orientation
                            # not in the considered range
                            continue
                        total += norm[cell_row_idx, cell_column_idx]
                orientation_histogram[r_i, c_i, i] =  \
                    total / (cell_rows * cell_columns)
                c_i += 1
                c += cell_columns
            r_i += 1
            r += cell_rows
    return orientation_histogram


def _normalize_all(orientation_histogram: np.ndarray,
                   cells_per_block: Tuple[int, int],
                   block_norm: str) -> np.ndarray:
    """Return array of all the normalized blocks from the orientation
    histogram.

    Parameters
    ----------
    orientation_histogram : np.ndarray
        Histogram of gradient orientations in the image.
    cells_per_block : Tuple[int, int]
        Number of cells per block.
    block_norm : str
        Method of block normalization.

    Returns
    -------
    np.ndarray
        Normalized blocks.
    """
    block_rows, block_cols = cells_per_block
    num_cells_row, num_cells_col, orientations = orientation_histogram.shape
    num_blocks_row = (num_cells_row - block_rows) + 1
    num_blocks_col = (num_cells_col - block_cols) + 1
    normalized_blocks = np.zeros(
        (num_blocks_row, num_blocks_col, block_rows, block_cols, orientations),
        dtype=np.float64)

    for r in range(num_blocks_row):
        for c in range(num_blocks_col):
            # Get each block
            block = orientation_histogram[
                r:r + block_rows, c:c + block_cols, :]
            # Normalize it
            normalized_blocks[r, c, :] = _normalize_block(
                block, method=block_norm)
    return normalized_blocks


def _hog(image: np.ndarray, orientations: int, block_norm: str,
         pixels_per_cell: Tuple[int, int], cells_per_block: Tuple[int, int]
         ) -> np.ndarray:
    """HOG pipeline.

    Parameters
    ----------
    image : np.ndarray
        Input image of shape (W, H, C).
    orientations : int
        Number of orientation bins.
    block_norm : str
        Method of block normalization.
    pixels_per_cell : Tuple[int, int]
        Size in pixels of a cell.
    cells_per_block : Tuple[int, int]
        Number of pixels in each block.

    Returns
    -------
    np.ndarray
        Flattened histogram of oriented gradients.
    """
    gradient_row, gradient_col = _gradient(image)
    norm = np.hypot(gradient_row, gradient_col)
    orientation = np.rad2deg(np.arctan2(gradient_row, gradient_col)) % 180

    orientation_histogram = _histograms(
        norm, orientation, orientations, pixels_per_cell)

    normalized_blocks = _normalize_all(
        orientation_histogram, cells_per_block, block_norm)
    return normalized_blocks.ravel()


class HOG(Base):
    def __init__(self, orientations: int = 9,
                 block_norm: str = 'L2-Hys',
                 pixels_per_cell: Tuple[int, int] = (2, 2),
                 cells_per_block: Tuple[int, int] = (1, 1)) -> None:
        """Processor to extract Histogram of Oriented Gradients of images.

        Parameters
        ----------
        orientations : int, optional
            Number of orientation bins, by default 9.
        block_norm : str, optional
            Block normalization method, by default 'L2-Hys'.
            Must be in {'L1', 'L1-sqrt', 'L2', 'L2-Hys'}.
        pixels_per_cell : Tuple[int, int], optional
            Size in pixels of a cell, by default (2, 2).
        cells_per_block : Tuple[int, int], optional
            Number of cells in each block, by default (1, 1).
        """
        super().__init__()
        self.orientations = orientations
        self.pixels_per_cell = pixels_per_cell
        self.cells_per_block = cells_per_block
        self.block_norm = block_norm

    @property
    def name(self):
        return 'hog'

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Compute the HOG.

        Parameters
        ----------
        X : np.ndarray
            Array of images of shape (N, W, H, C) or (W, H, C).

        Returns
        -------
        np.ndarray
            Computed HOG.

        Raises
        ------
        ValueError
            If X has not correct dimensions.
        """
        self.log.debug(f'Process images of shape {X.shape}.')
        if X.ndim == 3:
            out = _hog(X, self.orientations, self.block_norm,
                       self.pixels_per_cell, self.cells_per_block)
        elif X.ndim == 4:
            def worker(i): return _hog(
                X[i], self.orientations, self.block_norm,
                self.pixels_per_cell, self.cells_per_block)
            out = np.asarray(Parallel(n_jobs=-1)(delayed(worker)(i)
                                                 for i in range(len(X))))
        else:
            raise ValueError(
                'Input must be array of shape [N, W, H, C] or [W, H, C]')
        self.log.debug('Done.')
        return out
