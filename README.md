# CycleGAN
 
- Link data: https://drive.google.com/drive/folders/1NZC7FLBOLz9AVSQag-TZwsHSCR_e-mRh?usp=sharing 
- Link pre_train model: 
- Generator B to A: https://drive.google.com/file/d/1BoGQd1Q5lFHI7f9JOS0PAbUXxdAyCohL/view?usp=sharing 
- Genetator A to B: https://drive.google.com/file/d/1DVBdguv8TrSFghp3cD4gTngha2YjKV5x/view?usp=sharing
---
# Traning detail: 
### Hardware
- Gpu = True, 1 Gpu 3090ti
- Time consuming = 5 hours
- Result: horse -> zibra is easier than zibra -> horse
### Hyperparameter
- Epochs = 100
- Decay_epochs = 50 -> optimize the epochs while training
- Batch size = 8
- lr = 0.0002
- image size = 256 pixel
