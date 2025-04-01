# CheapMatting

**Probably the most efficient image matting network.**

CheapMatting is an efficient image matting network built on our ATLDAS NAS method, designed for real-time, high-quality matting applications. Leveraging our automatic topology learning approach for differentiable architecture search, this network achieves impressive performance on both synthetic and real-world scenarios.

## Key Features

- **Real-Time Performance:** Achieves 1080p image matting in real time at 26 FPS on an NVIDIA 2080Ti.
- **Low Memory Consumption:** Requires only 2 GiB of GPU memory.
- **High Accuracy:** Demonstrates an SAD error of 33 on the Adobe 1K dataset.
- **Real-World Ready:** We will provide a checkpoint (ckpt) optimized for real-world applications.

## Background

CheapMatting is constructed based on our ATLDAS (Automatic Topology Learning for Differentiable Architecture Search) method. By automatically learning optimal network topologies, we balance efficiency and accuracy, making it suitable for practical image matting tasks.

## Usage

### Environment Setup

- **Hardware:** NVIDIA 2080Ti (or equivalent) is recommended.
- **Dependencies:** Ensure you have the required Python libraries and CUDA properly installed.
- **Checkpoint:** Download the provided `adobe1k.ckpt` for real-world scenarios.

### Example Code

Below is a sample Python script to demonstrate how to load the model and perform image matting:

```python
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import cv2
import numpy as np
import torch
import model

# Define paths for trimap, merged images, and output predictions
p1 = './trimaps/'
p2 = './merged/'
p3a = './pred/'
os.makedirs(p3a, exist_ok=True)

if __name__ == '__main__':
    # Initialize and load the CheapMatting model
    segmodel = model.CheapMatting()
    segmodel.load_state_dict(torch.load('./adobe1k.ckpt', map_location='cpu')['model'])
    segmodel = segmodel.cuda()
    segmodel.eval()
    
    # Process each image in the trimaps directory
    for idx, file in enumerate(os.listdir(p1)):
        rawimg_path = p2 + file
        trimap_path = p1 + file
        
        rawimg = cv2.imread(rawimg_path)
        trimap = cv2.imread(trimap_path, cv2.IMREAD_GRAYSCALE)
        trimap_nonp = trimap.copy()
        h, w, c = rawimg.shape

        # Calculate padding to make dimensions a multiple of 32
        newh = (((h - 1) // 32) + 1) * 32
        neww = (((w - 1) // 32) + 1) * 32
        padh = newh - h
        padh1 = int(padh / 2)
        padh2 = padh - padh1
        padw = neww - w
        padw1 = int(padw / 2)
        padw2 = padw - padw1

        # Apply padding using reflection padding
        rawimg_pad = cv2.copyMakeBorder(rawimg, padh1, padh2, padw1, padw2, cv2.BORDER_REFLECT)
        trimap_pad = cv2.copyMakeBorder(trimap, padh1, padh2, padw1, padw2, cv2.BORDER_REFLECT)

        # Create a 3-channel trimap mask for the model
        tritemp = np.zeros([*trimap_pad.shape, 3], np.float32)
        tritemp[:, :, 0] = (trimap_pad == 0)
        tritemp[:, :, 1] = (trimap_pad == 128)
        tritemp[:, :, 2] = (trimap_pad == 255)
        tritemp2 = np.transpose(tritemp, (2, 0, 1))
        tritemp2 = tritemp2[np.newaxis, :, :, :]

        # Prepare the image data
        img = np.transpose(rawimg_pad, (2, 0, 1))[np.newaxis, ::-1, :, :]
        img = np.array(img, np.float32) / 255.
        img = torch.from_numpy(img).cuda()
        tritemp2 = torch.from_numpy(tritemp2).cuda()

        # Perform inference
        with torch.no_grad():
            all_data = torch.cat([img, tritemp2], 1)
            pred = segmodel(all_data)
            pred = pred.detach().cpu().numpy()[0]
            pred = pred[:, padh1:padh1+h, padw1:padw1+w]
            preda = pred[0:1, ] * 255
            preda = np.transpose(preda, (1, 2, 0))
            preda = preda * (trimap_nonp[:, :, None] == 128) + (trimap_nonp[:, :, None] == 255) * 255

        preda = np.array(preda, np.uint8)
        cv2.imwrite(p3a + file, preda)
```

## Citation

If you use CheapMatting in your work, please consider citing the following publications:

### Journal Article

```bibtex
@article{liu2023atldas,
  title = {ATL-DAS: Automatic Topology Learning for Differentiable Architecture Search},
  journal = {Displays},
  volume = {80},
  pages = {102541},
  year = {2023},
  author = {Qinglin Liu and Jingbo Lin and Xiaoqian Lv and Wei Yu and Zonglin Li and Shengping Zhang},
}

@inproceedings{liu2024aematter, 
  title = {Revisiting Context Aggregation for Image Matting},
  author = {Liu, Qinglin and Lv, Xiaoqian and Meng, Quanling and Li, Zonglin and Lan, Xiangyuan and Yang, Shuo and Zhang, Shengping and Nie, Liqiang},
  booktitle = {International Conference on Machine Learning (ICML)},
  year = {2024},
}
```

## Contact & Feedback
For any questions or feedback, please open an issue on GitHub or contact us via email. We look forward to collaborating and advancing image matting technology together!
