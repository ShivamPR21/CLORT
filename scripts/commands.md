# CLORT Run Shell Commands

### Fresh Run

```shell
python scripts/train.py 'dataset.workers=4' 'dataset.prefetch=4' wb=clort model=mv_enc dataset=argo_mv loss=basic mb=single_center val_mb=infer optimizer=optim
```

### Resume with `<ID> <Model> <n_epochs>`

```shell
python scripts/train.py 'dataset.workers=4' 'dataset.prefetch=4' 'wb.resume="must"' 'wb.run_id="<ID>"' 'model.restore=True' 'model.run_path="<ID Path>"' 'model.model_file="<Model>"' 'optimizer.n_epochs=<n_epoch>' wb=clort model=mv_enc dataset=argo_mv loss=basic mb=single_center val_mb=infer optimizer=optim
```

## MultiView Encoder

1. Model : **mv_enc**
2. Dataset: **argo_mv**
3. loss: **basic** / basic+static_contrast / basic+joint_tracks / basic+static_contrast+joint_tracks
4. mb: **single_center** / three_center / five_center
5. mb.init: **zeros** / uniform / orthogonal.uniform / orthogonal.distributed
6. mb.init_dilation: 1 / 3 / 5
7. mb.init_density: 1 / 3 / 5

## PointCloud Encoder

## MultiModal Encoder

## MultiModalContext Encoder
