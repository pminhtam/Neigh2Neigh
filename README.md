# Neighbor2Neighbor: Self-Supervised Denoising from Single Noisy Images

Paper https://arxiv.org/abs/2101.02824

# Train
```
python  train.py --noise_dir ../image/Noisy/ --gt_dir ../image/Clean/ --image_size 128 --batch_size 1 --save_every 1000 --loss_every 100 -nw 1 -c  -ckpt n2n
```


# Test 
```
python test_dual.py  -n /mnt/vinai/SIDD/ValidationNoisyBlocksSrgb.mat  -g /mnt/vinai/SIDD/ValidationGtBlocksSrgb.mat  -c -ckpt mir_kpn -m KPN
```

## Requirement 

