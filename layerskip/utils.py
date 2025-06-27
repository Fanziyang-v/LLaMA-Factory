# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


from typing import List


def slice_str_to_array(slice_str: str, length: int) -> List[bool]:
    """
    Convert a string representing a Python slice or index into a boolean array.

    The resulting array will have the same length as the specified `length` parameter.
    Each element in the array corresponds to an index in the original sequence,
    with `True` indicating that the index is included in the slice and `False` otherwise.

    Args:
        slice_str (str): A string representing a Python slice or index, e.g. "1:3", ":5", "2::3", "0,4,5".
        length (int): The length of the original sequence.

    Returns:
        List[bool]: A boolean array representing the slice.

    Examples:
        >>> slice_str_to_array("1:3", 5)
        [False, True, True, False, False]
        >>> slice_str_to_array(":", 5)
        [True, True, True, True, True]
        >>> slice_str_to_array("::2", 5)
        [True, False, True, False, True]
        >>> slice_str_to_array("1::2", 5)
        [False, True, False, True, False]
        >>> slice_str_to_array("2:5:2", 6)
        [False, False, True, False, True, False]
        >>> slice_str_to_array("0,4,5", 7)
        [True, False, False, False, True, True, False]
    """

    assert "," not in slice_str or ":" not in slice_str, "Cannot mix commas and colons"

    if "," in slice_str:
        indices = [int(i) for i in slice_str.split(",")]
        assert all(0 <= i < length for i in indices), "Index out of range"
        result = [False] * length
        for i in indices:
            result[i] = True
        return result

    parts = slice_str.split(":")
    assert len(parts) <= 3, "Invalid slice format"
    start, end, step = None, None, None

    if len(parts) == 1 and parts[0] != "":
        start = int(parts[0])
        end = start + 1
        step = 1
    elif len(parts) == 2:
        start = int(parts[0]) if parts[0] != "" else None
        end = int(parts[1]) if parts[1] != "" else None
    elif len(parts) == 3:
        start = int(parts[0]) if parts[0] != "" else None
        end = int(parts[1]) if parts[1] != "" else None
        step = int(parts[2]) if parts[2] != "" else None

    assert start is None or 0 <= start < length, "Start index out of range"
    assert end is None or 0 <= end < length, "End index out of range"
    assert step is None or step != 0, "Step cannot be zero"

    result = [False] * length
    slice_indices = range(
        start if start is not None else 0,
        end if end is not None else length,
        step if step is not None else 1,
    )

    for i in slice_indices:
        if 0 <= i < length:
            result[i] = True

    return result
