import torch
import torch.nn.functional as F
from torchvision import transforms
import os, numpy as np, cv2
from scipy.ndimage.filters import gaussian_filter


def rand_float_in_range(min_value, max_value):
	return (torch.rand((1,)).item() * (max_value - min_value)) + min_value


def random_dvf(shape, sigma=None, alpha=None):
	"""
	generates random dvf along given axis
	"""
	if sigma is None:
		sigma = rand_float_in_range(max(shape)//16, max(shape)//8)
	else:
		sigma = rand_float_in_range(sigma//2, sigma)

	if alpha is None:
		alpha = rand_float_in_range(0.01, 0.1)
	else:
		alpha = rand_float_in_range(0.01, alpha)

	g = gaussian_filter(torch.rand(*shape).numpy(), sigma, cval=0)
	g = ( (g / g.max()) * 2 - 1 ) * alpha
	g = torch.Tensor(g)
	return g


def random_gaussian(shape, grid, sigma=None, alpha=None):
	"""
	generates random gaussian field along given axis
	"""
	if sigma is None:
		sigma = rand_float_in_range(shape//8, shape//4)
	else:
		sigma = rand_float_in_range(sigma//2, sigma)

	if alpha is None:
		alpha = rand_float_in_range(-0.2, 0.2)
	else:
		alpha = rand_float_in_range(-alpha, alpha)

	if abs(alpha) < 0.02:
		alpha = 0.02

	center = rand_float_in_range(-0.99, 0.99)
	g = alpha * torch.exp(-((grid * shape - center)**2 / (2.0 * sigma**2)))
	print(alpha, g.max(), g.min())
	return g


class RandomElasticTransform(object):
	"""
	blah blah 
	"""

	def __init__(self, p=0.75, alpha=None, sigma=None):
		self.p = p
		self.alpha = alpha
		self.sigma = sigma

	def __call__(self, img, target=None):
		"""
		Args:
			img (Torch Tensor): Tensor image to be rotated. c*h*w
			target (Torch Tensor): optional target image to apply the same transformation to

		Returns:
			img: Randomly elastic transformed image.
			target: if target is not None
		"""

		if torch.rand((1,)).item() <= self.p:
			c, h, w = img.shape
			grid_y, grid_x = torch.meshgrid(torch.linspace(-1, 1, steps=h), torch.linspace(-1, 1, steps=w))

			dx = random_dvf((h, w), sigma=self.sigma, alpha=self.alpha) + random_gaussian(w, grid_x, self.sigma, self.alpha)
			dy = random_dvf((h, w), sigma=self.sigma, alpha=self.alpha) + random_gaussian(h, grid_y, self.sigma, self.alpha)
			indices = torch.stack([grid_x + dx, grid_y + dy])
			indices = indices.permute(1, 2, 0)

			img = F.grid_sample(img.view(1, c, h, w),
								indices.view(1, h, w, 2), mode="nearest")
			img = img.view(c, h, w)

			if target is not None:
				target = F.grid_sample(target.view(1, c, h, w),
									indices.view(1, h, w, 2), mode="nearest")
				target = target.view(c, h, w)

				return (img, target)

		return img

	def __repr__(self):
		return self.__class__.__name__ + '(p={0})'.format(self.p)


if __name__ == '__main__':
	os.makedirs("sanity", exist_ok=True)
	input1 = np.zeros((64, 64)).astype(np.float32)
	input1[10:50, 20:40] = 0.5

	data_generator = transforms.Compose([
		transforms.ToTensor(),
		RandomElasticTransform(alpha=1.0)
		])

	input2 = data_generator(input1)
	print(input2.shape)
	input2 = input2[0].numpy()
	cv2.imwrite("sanity/im1.jpg", (input1*255).astype(np.uint8))
	cv2.imwrite("sanity/im2.jpg", (input2*255).astype(np.uint8))



