import torch
import torch.nn as nn
import torch.nn.functional as F


def heat_1d_kernel(kernel_size, sigma):
    """Creates unidimensional heat kernel tensor
    Args:
        kernel_size (int): tensor size
        sigma (float): gaussian standard deviation
    Returns:
        type: torch.Tensor
    """
    centered_x = torch.linspace(0, kernel_size - 1, kernel_size).sub(kernel_size // 2)
    heat_1d_tensor = torch.exp(- centered_x.pow(2).div(float(2 * sigma**2)))
    heat_1d_tensor.div_(heat_1d_tensor.sum())
    return heat_1d_tensor


def heat_2d_kernel(kernel_size, channels):
    """Creates square bidimensional heat kernel tensor with specified number of
    channels
    Args:
        kernel_size (int): tensor size
        sigma (float): gaussian standard deviation
    Returns:
        type: torch.Tensor
    """
    _1d_window = heat_1d_kernel(kernel_size, 1.5).unsqueeze(1)
    _2d_window = _1d_window.mm(_1d_window.t()).float().unsqueeze(0).unsqueeze(0)
    kernel = _2d_window.expand(channels, 1, kernel_size, kernel_size).contiguous()
    return kernel


class SSIM(nn.Module):
    '''
    NOTE: This is a SIMILARITY metrics so if you are using it as a loss you are supposed
    to do 1-ssim so that you are maximizing the similarity when optimizing your model
    '''
    """Differential module implementing Structural-Similarity index computation
    "Image quality assessment: from error visibility to structural similarity",
    Wang et. al 2004
    Largely based on the work of https://github.com/Po-Hsun-Su/pytorch-ssim
    Args:
        kernel_size (int): convolutive kernel size
        C1 (float): weak denominator stabilizing constant (default: 0.01 ** 2)
        C2 (float): weak denominator stabilizing constant (default: 0.03 ** 2)
    """
    def __init__(self, kernel_size=11, C1=0.01**2, C2=0.03**2):
        super(SSIM, self).__init__()
        self.kernel_size = kernel_size
        self.channels = 1
        self.kernel = heat_2d_kernel(kernel_size, self.channels)
        self.C1 = C1
        self.C2 = C2

    def _compute_ssim(self, img1, img2, kernel):
        """Computes mean SSIM between two batches of images given convolution kernel
        Args:
            img1 (torch.Tensor): (B, C, H, W)
            img2 (torch.Tensor): (B, C, H, W)
            kernel (torch.Tensor): convolutive kernel used for moments computation
        Returns:
            type: torch.Tensor
        """
        # Retrieve number of channels and padding values
        channels = img1.size(1)
        padding = self.kernel_size // 2

        # Compute means tensors
        mu1 = F.conv2d(input=img1, weight=kernel, padding=padding, groups=channels)
        mu2 = F.conv2d(input=img2, weight=kernel, padding=padding, groups=channels)
        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1.mul(mu2)

        # Compute std tensors
        sigma1_sq = F.conv2d(input=img1 * img1, weight=kernel, padding=padding, groups=channels).sub(mu1_sq)
        sigma2_sq = F.conv2d(input=img2 * img2, weight=kernel, padding=padding, groups=channels).sub(mu2_sq)
        sigma12 = F.conv2d(input=img1 * img2, weight=kernel, padding=padding, groups=channels).sub(mu1_mu2)

        # Compute ssim map and return average value
        ssim_map = ((2 * mu1_mu2 + self.C1) * (2 * sigma12 + self.C2)) / ((mu1_sq + mu2_sq + self.C1) * (sigma1_sq + sigma2_sq + self.C2))
        return ssim_map.mean()

    def forward(self, img1, img2):
        """Computes mean SSIM between two batches of images
        Args:
            img1 (torch.Tensor): (B, C, H, W)
            img2 (torch.Tensor): (B, C, H, W)
        Returns:
            type: torch.Tensor
        """
        # If needed, recompute convolutive kernel
        channels = img1.size(1)
        if channels == self.channels and self.kernel.data.type() == img1.data.type():
            kernel = self.kernel
        else:
            kernel = heat_2d_kernel(self.kernel_size, channels)
            kernel = kernel.to(img1.device).type_as(img1)
            self.kernel = kernel
            self.channels = channels

        # Compute mean ssim
        ssim = self._compute_ssim(img1, img2, kernel)
        return ssim

    
class GANLoss(nn.Module):
    """Define different GAN objectives.
    The GANLoss class abstracts away the need to create the target label tensor
    that has the same size as the input.
    """

    def __init__(self, gan_mode='lsgan', target_real_label=1.0, target_fake_label=0.0):
        """ Initialize the GANLoss class.
        Parameters:
            gan_mode (str) - - the type of GAN objective. It currently supports vanilla, lsgan, and wgangp.
            target_real_label (bool) - - label for a real image
            target_fake_label (bool) - - label of a fake image
        Note: Do not use sigmoid as the last layer of Discriminator.
        LSGAN needs no sigmoid. vanilla GANs will handle it with BCEWithLogitsLoss.
        """
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.gan_mode = gan_mode
        if gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
        elif gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode in ['wgangp']:
            self.loss = None
        else:
            raise NotImplementedError('gan mode %s not implemented' % gan_mode)

    def get_target_tensor(self, prediction, target_is_real):
        """Create label tensors with the same size as the input.
        Parameters:
            prediction (tensor) - - tpyically the prediction from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images
        Returns:
            A label tensor filled with ground truth label, and with the size of the input
        """

        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction)

    def __call__(self, prediction, target_is_real):
        """Calculate loss given Discriminator's output and grount truth labels.
        Parameters:
            prediction (tensor) - - tpyically the prediction output from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images
        Returns:
            the calculated loss.
        """
        if self.gan_mode in ['lsgan', 'vanilla']:
            target_tensor = self.get_target_tensor(prediction, target_is_real)
            loss = self.loss(prediction, target_tensor)
        elif self.gan_mode == 'wgangp':
            if target_is_real:
                loss = -prediction.mean()
            else:
                loss = prediction.mean()
        return loss