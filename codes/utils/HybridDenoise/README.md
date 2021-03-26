# Single-frame Hybrid Noise Reduction 
This is the prototype code for the Hybrid Noise Reduction (ISP-HNR)

## Setup Dependencies
```console
$ pip install -r requirements.txt
```

## Run batch test (Latest)
```console
python run_hnr_batch.py --input_dir /Users/hgc/Downloads/night_samples_0115/*.jpg --nr_method median --nr_parm 7 -no_blend --mask_level -2
```

## Run single test samples
Recommended setting 1:

Frequency domain (wavelet) desnoising using sigma of 0.05 blend mask computed at
the second last Laplician pyramid.
```console
python hnr.py --input_fn imgs/noisy1.jpg --nr_method freq --nr_parm 0.05 --mask_level -2 
```

Fast NLM desnoising with neighborhood size of 5 blend mask computed at
the second last Laplician pyramid.

Recommended setting 2:
```console
python hnr.py --input_fn imgs/noisy1.jpg --nr_method NLM --nr_parm 5 --mask_level -2 
```