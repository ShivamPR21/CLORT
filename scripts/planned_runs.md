# Planned Runs

## Complted

1. ```python scripts/train.py wb=clortjsr model=mv_enc dataset=argo_mv loss=basic mb=single_center val_mb=infer optimizer=optim``` **Loss -> basic | MB -> single_center | Model -> MV** [WB-ID: "wdo6ekmv"](shivampr21/CLORTJSR/wdo6ekmv)
2. ```python scripts/train.py wb=clortjsr model=mv_enc dataset=argo_mv loss=basic mb=three_center val_mb=infer optimizer=optim``` **Loss -> basic | MB -> three_center | Model -> MV** [WB-ID: "yi2wjojk"](shivampr21/CLORTJSR/yi2wjojk)
3. ```python scripts/train.py wb=clortjsr model=mv_enc dataset=argo_mv loss=basic mb=five_center val_mb=infer optimizer=optim``` **Loss -> basic | MB -> five_center | Model -> MV** [WB-ID: "w0ulc279"](shivampr21/CLORTJSR/w0ulc279)
4. ```python scripts/train.py wb=clortjsr model=mv_enc dataset=argo_mv loss=basic mb=three_center_orthogonal_uniform val_mb=infer optimizer=optim``` **Loss -> basic | MB -> three_center_orthogonal_uniform | Model -> MV** [WB-ID: "kif096f0"](shivampr21/CLORTJSR/kif096f0)
5. ```python scripts/train.py wb=clortjsr model=mv_enc dataset=argo_mv loss=basic mb=three_center_orthogonal_distributed val_mb=infer optimizer=optim``` **Loss -> basic | MB -> three_center_orthogonal_distributed | Model -> MV** [WB-ID: "bc9yx4zk"](shivampr21/CLORTJSR/bc9yx4zk)
6. ```python scripts/train.py wb=clortjsr model=mv_enc dataset=argo_mv loss=loss_v1 mb=five_center val_mb=infer optimizer=optim``` **Loss -> loss_v1 | MB -> five_center | Model -> MV** [WB-ID: "7fywl9ia"](shivampr21/CLORTJSR/7fywl9ia)
7. ```python scripts/train.py wb=clortjsr model=mv_enc dataset=argo_mv loss=loss_v2 mb=five_center val_mb=infer optimizer=optim``` **Loss -> loss_v2 | MB -> five_center | Model -> MV** [WB-ID: "psc7fvlb"](shivampr21/CLORTJSR/psc7fvlb)
8. ```python scripts/train.py wb=clortjsr model=mm_enc dataset=argo loss=loss_v2 mb=five_center val_mb=infer optimizer=optim restore=none``` **Loss -> loss_v2 | MB -> five_center | Model -> MM** [WB-ID: "r4722tp1"](shivampr21/CLORTJSR/r4722tp1)
9. Base frozen training of MMC encoder on MM fine tuned model "pbi1ca5m" "model_20.pth": ```python scripts/train.py 'model.restore=True' 'model.model_file=model_20.pth' 'model.run_path="shivampr21/CLORTJSR/pbi1ca5m"' wb=clortjsr model=mmc_enc dataset=argo loss=loss_v2_1 mb=five_center val_mb=infer optimizer=optim_base_frozen restore=model``` **Loss -> loss_v2_1 | MB -> five_center | Model -> MMC** [WB-ID: "45xt24wt"](shivampr21/CLORTJSR/45xt24wt)
10. ```python scripts/train.py 'loss.t_inc_coeff=0.001' wb=clortjsr model=mv_xo_enc dataset=argo_mv loss=loss_v2_1 mb=five_center val_mb=infer optimizer=optim restore=none``` **Loss -> loss_v2 | MB -> five_center | Model -> MV-XO** [WB-ID: "saa7dijm"](shivampr21/CLORTJSR/saa7dijm)

## Extended Run

1. ```python scripts/train.py 'wb.resume="must"' 'wb.run_id="itkorpud"' 'model.restore=True' 'model.model_file=model_10.pth' 'model.run_path="shivampr21/CLORTJSR/itkorpud"' wb=clortjsr model=mm_enc dataset=argo loss=loss_v2 mb=five_center val_mb=infer optimizer=extended_run_20``` **Loss -> loss_v2 | MB -> five_center | Model -> MM** [WB-ID: "itkorpud"](shivampr21/CLORTJSR/itkorpud)
2. Extended Separate run on "r4722tp1": ```python scripts/train.py 'model.restore=True' 'model.model_file=model_10.pth' 'model.run_path="shivampr21/CLORTJSR/r4722tp1"' 'loss.t_inc_coeff=0.0005' wb=clortjsr model=mm_enc dataset=argo loss=loss_v2_1 mb=five_center val_mb=infer optimizer=extended_run_20 restore=complete_no_t``` **Loss -> loss_v2_1 | MB -> five_center | Model -> MM** [WB-ID: "pbi1ca5m"](shivampr21/CLORTJSR/pbi1ca5m)

## Deprecated

1. ```python scripts/train.py wb=clortjsr model=mm_enc dataset=argo loss=loss_v2 mb=five_center val_mb=infer optimizer=optim``` **Loss -> loss_v2 | MB -> five_center | Model -> MM** [WB-ID: "itkorpud"](shivampr21/CLORTJSR/itkorpud)
2. ```python scripts/train.py 'model.restore=True' 'model.model_file=model_20.pth' 'model.run_path="shivampr21/CLORTJSR/itkorpud"' wb=clortjsr model=mmc_enc dataset=argo loss=loss_v2 mb=five_center val_mb=infer optimizer=optim_base_frozen restore=model``` **Loss -> loss_v2 | MB -> five_center | Model -> MMC** [WB-ID: "7t8sm6j8"](shivampr21/CLORTJSR/7t8sm6j8)

## Running

1. MV frozen training of MM-XO encoder on MV-XO model "saa7dijm" "model_10.pth": ```python scripts/train.py 'model.restore=True' 'model.model_file=model_10.pth' 'model.run_path="shivampr21/CLORTJSR/saa7dijm"' 'loss.t_inc_coeff=0.001' wb=clortjsr model=mm_xo_enc dataset=argo loss=loss_v2_1 mb=five_center val_mb=infer optimizer=optim_mv_frozen restore=model``` **Loss -> loss_v2_1 | MB -> five_center | Model -> MM-XO** [WB-ID: "zzw260si"](shivampr21/CLORTJSR/zzw260si)
2. MM frozen training of MMC-XO encoder on MV model "zzw260si" "model_10.pth": ```python scripts/train.py 'model.restore=True' 'model.model_file=model_10.pth' 'model.run_path="shivampr21/CLORTJSR/zzw260si"' 'loss.t_inc_coeff=0.001' wb=clortjsr model=mmc_xo_enc dataset=argo loss=loss_v2_1 mb=five_center val_mb=infer optimizer=optim_mm_frozen restore=model``` **Loss -> loss_v2_1 | MB -> five_center | Model -> MM-XO* [WB-ID: ""](shivampr21/CLORTJSR/)

## Planned

1. ```python scripts/train.py wb=clortjsr model=mv_enc dataset=argo_mv loss=loss_v1 mb=three_center val_mb=infer optimizer=optim``` Loss -> loss_v1 | MB -> three_center | Model -> MV
2. ```python scripts/train.py wb=clortjsr model=mv_enc dataset=argo_mv loss=loss_v2 mb=three_center val_mb=infer optimizer=optim``` Loss -> loss_v2 | MB -> three_center | Model -> MV
3. ```python scripts/train.py wb=clortjsr model=mm_enc dataset=argo loss=loss_v1 mb=five_center val_mb=infer optimizer=optim``` Loss -> loss_v1 | MB -> five_center | Model -> MM
4. ```python scripts/train.py wb=clortjsr model=mmc_enc dataset=argo loss=loss_v1 mb=five_center val_mb=infer optimizer=optim``` Loss -> loss_v1 | MB -> five_center | Model -> MMC
5. ```python scripts/train.py wb=clortjsr model=mv_xo_enc dataset=argo loss=loss_v1 mb=five_center val_mb=infer optimizer=optim``` Loss -> loss_v1 | MB -> five_center | Model -> MV-XO
6. ```python scripts/train.py wb=clortjsr model=mm_xo_enc dataset=argo loss=loss_v1 mb=five_center val_mb=infer optimizer=optim``` Loss -> loss_v1 | MB -> five_center | Model -> MM-XO
7. ```python scripts/train.py wb=clortjsr model=mmc_xo_enc dataset=argo loss=loss_v1 mb=five_center val_mb=infer optimizer=optim``` Loss -> loss_v1 | MB -> five_center | Model -> MMC-XO

## Templates

- Fresh Run: ```python scripts/train.py wb=clortjsr model=mmc_enc dataset=argo loss=loss_v2 mb=five_center val_mb=infer optimizer=optim``` *Loss -> loss_v2 | MB -> five_center | Model -> MMC* [WB-ID: ""](shivampr21/CLORTJSR/)
- Frozen Run: ```python scripts/train.py 'model.restore=True' 'model.model_file=model_10.pth' 'model.run_path="shivampr21/CLORTJSR/o3v6diy8"' wb=clortjsr model=mmc_enc dataset=argo loss=loss_v2 mb=five_center val_mb=infer optimizer=optim_base_frozen``` *Loss -> loss_v2 | MB -> five_center | Model -> MMC* [WB-ID: ""](shivampr21/CLORTJSR/)
- Extended Run: ```python scripts/train.py 'wb.resume="must"' 'wb.run_id="o3v6diy8"' 'model.restore=True' 'model.model_file=model_10.pth' 'model.run_path="shivampr21/CLORTJSR/o3v6diy8"' wb=clortjsr model=mm_enc dataset=argo loss=loss_v2 mb=five_center val_mb=infer optimizer=extended_run_15``` *Loss -> loss_v2 | MB -> five_center | Model -> MM* [WB-ID: ""](shivampr21/CLORTJSR/)

## Scrapped

- ```python scripts/train.py wb=clortjsr model=mmc_enc dataset=argo loss=loss_v2 mb=five_center val_mb=infer optimizer=optim``` *Loss -> loss_v2 | MB -> five_center | Model -> MMC* [WB-ID: "t8yc99hv"](shivampr21/CLORTJSR/t8yc99hv)
