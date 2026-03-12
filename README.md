<!--
SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
SPDX-License-Identifier: Apache-2.0
SPDX-PackageName: senpai
-->

## Dev Environment: Devpod

The devpod has global python installed, so you can directly run scripts with `python my_script.py`, no need to use `uv` or `venv`.

1. Start the devbox:
```bash
devpod up . --id senpai
```

If the pod does not schedule, set the provider options once (CPU allocatable is slightly under 128 on the 8x GPU nodes, so we cap at 120). Also ensure the pod template mounts `/dev/shm`:
```bash
devpod provider set-options cw-cfd -o POD_MANIFEST_TEMPLATE=.devcontainer/pod-template.yaml
devpod provider set-options cw-cfd -o RESOURCES=limits.nvidia.com/gpu=8,requests.nvidia.com/gpu=8,limits.cpu=120,requests.cpu=120,limits.memory=960Gi,requests.memory=960Gi
```
The pod template adds a 32Gi `/dev/shm` mount for the devpod container.

2. Stop the devbox

Stop when not in use to release GPU resources:
```bash
devpod stop senpai
```
This deletes the pod (frees GPU) but keeps your data. Run `devpod up` again to resume.

To delete everything (pod + data):
```bash
devpod delete senpai
```

## References
`TandemFoilSet: Datasets for Flow Field Prediction of Tandem-Airfoil Through the Reuse of Single Airfoils`
is distributed by CC-BY-4.0.
```bibtex
@inproceedings{
lim2026tandemfoilset,
title={**TandemFoilSet**: Datasets for Flow Field Prediction of Tandem-Airfoil Through the Reuse of Single Airfoils},
author={Wei Xian Lim and Loh Sher En Jessica and Zenong Li and Thant Zin Oo and Wai Lee Chan and Adams Wai-Kin Kong},
booktitle={The Fourteenth International Conference on Learning Representations},
year={2026},
url={https://openreview.net/forum?id=4Z0P4Nbosn}
}
```
