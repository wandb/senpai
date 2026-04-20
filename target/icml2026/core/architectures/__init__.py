# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai

"""Reference and Senpai model variants for the ICML 2026 sprint."""

from core.architectures.abupt_reference import ABUPTReference
from core.architectures.senpai_transolver import ANPSurfaceDecoder, SenpaiTransolver
from core.architectures.transolver_reference import ReferenceTransolver

__all__ = ["ABUPTReference", "ANPSurfaceDecoder", "ReferenceTransolver", "SenpaiTransolver"]
