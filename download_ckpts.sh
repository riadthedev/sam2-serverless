#!/bin/bash

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


# Define the URL for the large checkpoint only
BASE_URL="https://dl.fbaipublicfiles.com/segment_anything_2/072824/"
sam2_hiera_l_url="${BASE_URL}sam2_hiera_large.pt"

echo "Downloading sam2_hiera_large.pt checkpoint..."
wget -q --show-progress --progress=dot:mega "$sam2_hiera_l_url" || { echo "Failed to download checkpoint from $sam2_hiera_l_url"; exit 1; }

echo "Large checkpoint downloaded successfully."
