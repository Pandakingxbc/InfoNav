# Third-party software and assets

InfoNav depends on software, model checkpoints, and datasets maintained by other projects. The top-level GPL-3.0 license applies only to original InfoNav code unless a file or component states otherwise.

| Component | Location or use | Upstream terms |
| --- | --- | --- |
| D-FINE | `vlm/detector/D-FINE` submodule | Apache-2.0; see the submodule `LICENSE` |
| LKH-3.0.6 | `src/planner/utils/lkh_mtsp_solver` | The included upstream README says research use and that the author reserves all rights. This is not GPL-3.0. Review the upstream terms before redistribution or non-research use. |
| Habitat-Lab / Habitat-Sim | external dependency | See the [Habitat repository](https://github.com/facebookresearch/habitat-lab) |
| GroundingDINO | external dependency and checkpoint | See the [GroundingDINO repository](https://github.com/IDEA-Research/GroundingDINO) |
| BLIP-2 / LAVIS | external dependency and checkpoint | See the [LAVIS repository](https://github.com/salesforce/LAVIS) |
| MobileSAM | external dependency and checkpoint | See the [MobileSAM repository](https://github.com/ChaoningZhang/MobileSAM) |
| YOLOv7 | external dependency and checkpoint | See the [YOLOv7 repository](https://github.com/WongKinYiu/yolov7) |
| HM3D | scene and ObjectNav datasets | Registration/access terms apply; see [Habitat datasets](https://github.com/facebookresearch/habitat-lab/blob/v0.3.1/DATASETS.md) |
| MP3D | scene and ObjectNav datasets | Matterport access terms apply; see [Habitat datasets](https://github.com/facebookresearch/habitat-lab/blob/v0.3.1/DATASETS.md) |

Model weights and datasets are intentionally excluded from Git. Downloading a file does not change its upstream license. When publishing derived artifacts, cite the original projects and record the exact version or checkpoint used.

## Release note

Before creating an archival release, the maintainers should obtain legal confirmation that redistributing the bundled LKH source is permitted. If not, remove that source from the release and provide a user-side download/setup step that preserves the upstream terms.
