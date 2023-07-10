# Planned Runs

1. ```python scripts/train.py wb=clortjsr model=mv_enc dataset=argo_mv loss=basic mb=single_center val_mb=infer optimizer=optim``` **Loss -> basic | MB -> single_center | Model -> MV** [WB-ID: "wdo6ekmv"](shivampr21/CLORTJSR/wdo6ekmv)
2. ```python scripts/train.py wb=clortjsr model=mv_enc dataset=argo_mv loss=basic mb=three_center val_mb=infer optimizer=optim``` **Loss -> basic | MB -> three_center | Model -> MV** [WB-ID: "yi2wjojk"](shivampr21/CLORTJSR/yi2wjojk)
3. ```python scripts/train.py wb=clortjsr model=mv_enc dataset=argo_mv loss=basic mb=five_center val_mb=infer optimizer=optim``` **Loss -> basic | MB -> five_center | Model -> MV** [WB-ID: "w0ulc279"](shivampr21/CLORTJSR/w0ulc279)
4. ```python scripts/train.py wb=clortjsr model=mv_enc dataset=argo_mv loss=loss_v1 mb=three_center_orthogonal_uniform val_mb=infer optimizer=optim``` **Loss -> loss_v1 | MB -> three_center_orthogonal_uniform | Model -> MV** [WB-ID: "kif096f0"](shivampr21/CLORTJSR/kif096f0)
5. ```python scripts/train.py wb=clortjsr model=mv_enc dataset=argo_mv loss=loss_v1 mb=three_center_orthogonal_distributed val_mb=infer optimizer=optim``` Loss -> loss_v1 | MB -> three_center_orthogonal_distributed | Model -> MV
6. ```python scripts/train.py wb=clortjsr model=mv_enc dataset=argo_mv loss=loss_v1 mb=three_center val_mb=infer optimizer=optim``` Loss -> loss_v1 | MB -> three_center | Model -> MV
7. ```python scripts/train.py wb=clortjsr model=mv_enc dataset=argo_mv loss=loss_v2 mb=three_center val_mb=infer optimizer=optim``` Loss -> loss_v2 | MB -> three_center | Model -> MV
8. ```python scripts/train.py wb=clortjsr model=mm_enc dataset=argo loss=loss_v1 mb=three_center val_mb=infer optimizer=optim``` Loss -> loss_v1 | MB -> three_center | Model -> MM
9. ```python scripts/train.py wb=clortjsr model=mm_enc dataset=argo loss=loss_v1 mb=five_center val_mb=infer optimizer=optim``` Loss -> loss_v1 | MB -> five_center | Model -> MM
10. ```python scripts/train.py wb=clortjsr model=mmc_enc dataset=argo loss=loss_v1 mb=five_center val_mb=infer optimizer=optim``` Loss -> loss_v1 | MB -> five_center | Model -> MMC
11. ```python scripts/train.py wb=clortjsr model=mv_xo_enc dataset=argo loss=loss_v1 mb=five_center val_mb=infer optimizer=optim``` Loss -> loss_v1 | MB -> five_center | Model -> MV-XO
12. ```python scripts/train.py wb=clortjsr model=mm_xo_enc dataset=argo loss=loss_v1 mb=five_center val_mb=infer optimizer=optim``` Loss -> loss_v1 | MB -> five_center | Model -> MM-XO
13. ```python scripts/train.py wb=clortjsr model=mmc_xo_enc dataset=argo loss=loss_v1 mb=five_center val_mb=infer optimizer=optim``` Loss -> loss_v1 | MB -> five_center | Model -> MMC-XO
