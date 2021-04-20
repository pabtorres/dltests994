import torchvision.transforms.functional as TF
import torchvision
import torchvision.transforms as transforms

def range_slicing_mask(dimension_list, alpha, beta, gamma, delta):
  t = torch.ones(dimension_list)
  l1 = list(range(alpha)) + list(range(beta, dimension_list[0]))
  l2 = list(range(gamma)) + list(range(delta, dimension_list[0]))
  h = torch.tensor(l1)
  v = torch.tensor(l2)
  t.index_fill_(0, h, 0)
  t.index_fill_(1, v, 0)
  return t

def downsampling(in_tensor, scale):
  preprocess = transforms.Compose([
      transforms.Resize(scale),
      transforms.Resize(32)])
  return preprocess(in_tensor)

def color_jitter(in_tensor, brightness_factor, contrast_factor, saturation_factor, hue_factor):
  out_tensor = TF.adjust_brightness(in_tensor, brightness_factor)
  out_tensor = TF.adjust_contrast(out_tensor, contrast_factor)
  out_tensor = TF.adjust_saturation(out_tensor, saturation_factor)
  out_tensor = TF.adjust_hue(out_tensor, hue_factor)
  return out_tensor