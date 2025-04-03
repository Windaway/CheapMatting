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
            pred = pred[:, padh1:padh1 + h, padw1:padw1 + w]
            preda = pred[0:1, ] * 255
            preda = np.transpose(preda, (1, 2, 0))
            preda = preda * (trimap_nonp[:, :, None] == 128) + (trimap_nonp[:, :, None] == 255) * 255

        preda = np.array(preda, np.uint8)
        cv2.imwrite(p3a + file, preda)