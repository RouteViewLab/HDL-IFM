
import torch

from .superpoint import SuperPoint
from .superglue import SuperGlue
import time


class Matching(torch.nn.Module):
	""" Image Matching Frontend (SuperPoint + SuperGlue) """
	def __init__(self, config={}):
		super().__init__()
		self.superpoint = SuperPoint(config.get('superpoint', {}))
		self.superglue = SuperGlue(config.get('superglue', {}))

	def forward(self, data):
		""" Run SuperPoint (optionally) and SuperGlue
		SuperPoint is skipped if ['keypoints0', 'keypoints1'] exist in input
		Args:
		  data: dictionary with minimal keys: ['image0', 'image1']
		"""
		t1 = time.time()
		pred = {}

		# Extract SuperPoint (keypoints, scores, descriptors) if not provided
		
		if 'keypoints0' not in data:
			pred0 = self.superpoint({'image': data['image0']})
			pred = {**pred, **{k+'0': v for k, v in pred0.items()}}
		if 'keypoints1' not in data:
			pred1 = self.superpoint({'image': data['image1']})
			pred = {**pred, **{k+'1': v for k, v in pred1.items()}}
		
		# Batch all features
		# We should either have i) one image per batch, or
		# ii) the same number of local features for all images in the batch.
		data = {**data, **pred}

		for k in data:
			if isinstance(data[k], (list, tuple)):
				data[k] = torch.stack(data[k])

		# Perform the matching
		pred2 = self.superglue(data)
		#pred2 = {}
		pred = {**pred, **pred2}
		pred.pop('nokp_flag')
		pred = {k: v[0].cpu().numpy() for k, v in pred.items()}
		t2 = time.time()
		print("sp + superglue time {}".format(t2-t1))
		#pred = {**pred, **self.superglue(pred)}

		return pred
