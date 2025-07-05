import torch
from torch import Tensor
import torch.nn.functional as F

def dice_coeff(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon: float = 1e-6):
    # Average of Dice coefficient for all batches, or for a single mask
    assert input.size() == target.size()
    assert input.dim() == 3 or not reduce_batch_first

    sum_dim = (-1, -2) if input.dim() == 2 or not reduce_batch_first else (-1, -2, -3)

    inter = 2 * (input * target).sum(dim=sum_dim)
    sets_sum = input.sum(dim=sum_dim) + target.sum(dim=sum_dim)
    sets_sum = torch.where(sets_sum == 0, inter, sets_sum)

    dice = (inter + epsilon) / (sets_sum + epsilon)
    return dice.mean()


def multiclass_dice_coeff(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon: float = 1e-6):
    # Average of Dice coefficient for all classes
    return dice_coeff(input.flatten(0, 1), target.flatten(0, 1), reduce_batch_first, epsilon)


#def dice_loss(input: Tensor, target: Tensor, multiclass: bool = False):
    # Dice loss (objective to minimize) between 0 and 1
 #   fn = multiclass_dice_coeff if multiclass else dice_coeff
  #  return 1 - fn(input, target, reduce_batch_first=True)
#Multiklassen segmentation input und target haben nicht das gleiche Shape



def dice_loss(input: torch.Tensor, target: torch.Tensor, multiclass: bool = False, ignore_index: int = 255):
    """
    input: shape [B, C, H, W] (logits)
    target: shape [B, H, W] (class indices)
    """
    if target.dim() == 4:
        target = target.squeeze(1)

    # Maske für gültige Pixel (nicht ignore_index)
    valid_mask = target != ignore_index

    input = input.permute(0, 2, 3, 1)  # -> [B, H, W, C]
    input = input[valid_mask]          # nur gültige Pixel behalten
    target = target[valid_mask]        # dito

    if target.numel() == 0:
        return torch.tensor(0.0, device=input.device, requires_grad=True)

    num_classes = input.shape[-1]
    input_soft = F.softmax(input, dim=-1)
    target_one_hot = F.one_hot(target, num_classes=num_classes).float()

    intersection = (input_soft * target_one_hot).sum()
    union = input_soft.sum() + target_one_hot.sum()
    dice = 2. * intersection / (union + 1e-6)

    return 1 - dice