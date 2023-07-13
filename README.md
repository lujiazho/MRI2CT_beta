# MRI2CT_beta

Adapted from [ControlNet](https://github.com/lllyasviel/ControlNet) for MRI to CT task.

# Run

- git
```
git clone
cd MRI2CT_beta
```

- copy data config file to local
```
mkdir data
cp /ifs/loni/faculty/shi/spectrum/Student_2020/lzhong/MRI2CT/data/image_pairs.json ./data
```

The Dataset will directly refer to the prepared data under my directory, so it's safe to just run
```
CUDA_VISIBLE_DEVICES=3 python train_MRI2CT.py
```
