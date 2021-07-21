import torch, torchvision
import torchvision.transforms.functional as F

from PIL import Image, ImageOps


def cutout(img, pad_size, replace):
    img = F.pil_to_tensor(img)
    _, h, w = img.shape
    center_h, center_w = torch.randint(high=h, size=(1,)), torch.randint(high=w, size=(1,))
    low_h, high_h = torch.clamp(center_h-pad_size, 0, h).item(), torch.clamp(center_h+pad_size, 0, h).item()
    low_w, high_w = torch.clamp(center_w-pad_size, 0, w).item(), torch.clamp(center_w+pad_size, 0, w).item()
    cutout_img = img.clone()
    cutout_img[:, low_h:high_h, low_w:high_w] = replace
    return F.to_pil_image(cutout_img)


class Cutout(torch.nn.Module):
    """
    Apply cutout to the image.
    This operation applies a (2*pad_size, 2*pad_size) mask of zeros to a random location within image.
    The pixel values filled in will be of the value replace.
    """
    def __init__(self, p, pad_size, replace=128):
        super().__init__()
        self.p = p
        self.pad_size = int(pad_size)
        self.replace = replace

    def forward(self, image):
        if torch.rand(1) < self.p:
            cutout_image = cutout(image, self.pad_size, self.replace)
            return cutout_image
        else:
            return image

    def __repr__(self):
        return self.__class__.__name__ + "(p={0}, pad_size={1})".format(self.p, self.pad_size)
