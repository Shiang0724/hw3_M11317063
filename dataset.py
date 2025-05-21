import os
import torch
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np

class ETTSegmentationDataset(Dataset):
    def __init__(self, images_dir, masks_dir, transform=None):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.image_names = sorted(os.listdir(images_dir))
        self.mask_names = sorted(os.listdir(masks_dir))
        self.transform = transform

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        img_path = os.path.join(self.images_dir, self.image_names[idx])
        mask_path = os.path.join(self.masks_dir, self.mask_names[idx])

        # 讀取灰階影像
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        # ✅ 將灰階影像擴展為 3 通道 (符合 ResNet 預訓練模型輸入)
        image = np.repeat(image[np.newaxis, :, :], 3, axis=0)  # (3, H, W)
        mask = np.expand_dims(mask, axis=0)  # (1, H, W)

        # 正規化
        image = torch.tensor(image / 255., dtype=torch.float32)
        mask = torch.tensor(mask / 255., dtype=torch.float32)
        mask = (mask > 0.5).float()
        if self.transform:
            image, mask = self.transform(image, mask)

        return image, mask


def get_loaders(data_dir, fold, batch_size=16, shuffle=True):
    train_images = os.path.join(data_dir, fold, 'train')
    train_masks = os.path.join(data_dir, fold, 'trainannot')

    val_images = os.path.join(data_dir, fold, 'val')
    val_masks = os.path.join(data_dir, fold, 'valannot')

    train_dataset = ETTSegmentationDataset(train_images, train_masks)
    val_dataset = ETTSegmentationDataset(val_images, val_masks)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader


# 測試 DataLoader 是否正確
if __name__ == '__main__':
    train_loader, val_loader = get_loaders('processed_data', 'Fold1')

    print(f'Train batches: {len(train_loader)}')
    print(f'Validation batches: {len(val_loader)}')

    for images, masks in train_loader:
        print(f'Image shape: {images.shape}')  # ✅ 預期為 [batch_size, 3, 256, 256]
        print(f'Mask shape: {masks.shape}')    # ✅ 預期為 [batch_size, 1, 256, 256]
        break
