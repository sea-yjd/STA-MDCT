# STA-MDCT
<!-- This repository contains the code for STA-MDCT, a Pytorch library for adversarial machine learning research on speaker recognition. -->
This repository contains the code for STA-MDCT, a Pytorch library for security research on speaker recognition.

Paper: [STA-MDCT Paper](https://ieeexplore.ieee.org/document/10426806)

Website: [STA-MDCT Website](https://sea-yjd.github.io/)

Feel free to use SpeakerGuard for academic purpose 😄. For commercial purpose, please contact us 📫.

Cite our paper as follow:
```
@ARTICLE{STA-MDCT,
        author={Yao, Jiadi and Luo, Hong and Qi, Jun and Zhang, Xiao-Lei},
        journal={IEEE/ACM Transactions on Audio, Speech, and Language Processing}, 
        title={Interpretable Spectrum Transformation Attacks to Speaker Recognition Systems}, 
        year={2024},
        volume={32},
        number={},
        pages={1531-1545},
        keywords={Closed box;Glass box;Speaker recognition;Time-frequency analysis;Perturbation methods;Optimization;Data models;Speaker recognition;adversarial examples;adversarial transferability;black-box attacks},
        doi={10.1109/TASLP.2024.3364100}}
```

# 1. Usage
## 1.1 Requirements
pytorch=1.12.1, torchaudio=0.12.1, numpy=1.22.4, scipy=1.8.0

## 1.2 Dataset Preparation

### Attack to ASV
You can download them using the links: [voxceleb1_trials](https://www.robots.ox.ac.uk/~vgg/data/voxceleb/meta/veri_test.txt)

### Attack to SI
We provide five datasets, namely, Spk10_enroll, Spk10_test, Spk10_imposter, Spk251_train and Spk_251_test. You can manually download them using the follwing links:

[Spk10_enroll.tar.gz, 18MB, MD5:0e90fb00b69989c0dde252a585cead85](https://drive.google.com/uc?id=1BBAo64JOahk0F3yBAovnRLZ1NvjwBy7y&export\=download)

[Spk10_test.tar.gz, 114MB, MD5:b0f8eb0db3d2eca567810151acf13f16](https://drive.google.com/uc?id=1WctqJtP5Es74-U7y3cFXqfHi7JkDz6g5&export\=download)

[Spk10_imposter.tar.gz, 212MB, MD5:42abd80e27b78983a13b74e44a67be65](https://drive.google.com/uc?id=1f1GULs0aj_Xrw8JRxe6zzvTN3r2nnOf6&export\=download)

[Spk251_train.tar.gz, 10GB, MD5:02bee7caf460072a6fc22e3666ac2187](https://drive.google.com/uc?id=1iGcMPiPMzcCLI7xKJLwH1L0Ff_95-tmB&export\=download)

[Spk251_test.tar.gz, 1GB, MD5:182dd6b17f8bcfed7a998e1597828ed6](https://drive.google.com/uc?id=1rsXzuEyi5Zqd1XAsr1_Op7mC7hqY0tsp&export\=download)

After downloading, untar them inside `.Attack_for_SI/data` directory.

# 2. Attack
## 2.1 ASV

  ```
  python STA-MDCT/Attack_for_ASV/Attack/adversarial_attack.py
  ```
  In `Attack_for_ASV/Attack/attack_algorithm.py` file, we implemented the attack on a variety of gradient descent algorithms, including FGSM, PGD, CW, MI-FGSM, NI-FGSM, ACG, STA_MDCT.

## 2.2 SI (CSI/OSI)
```
  python STA-MDCT/Attack_for_SI/attackMain.py
  ```
  The calculation method of threshold (OSI) is in `Attack_for_SI/set_threshold.py`
# 3.CAM visualize
we propose to visualize the saliency maps of adversarial examples via the class activation maps (CAM). Our paper discusses Layer-CAM.

```
  python STA-MDCT/Attack_for_SI/LayerCAM_visualize.py
  ```
