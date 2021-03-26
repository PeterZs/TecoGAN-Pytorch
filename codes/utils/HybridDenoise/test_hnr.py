import unittest
from hnr import HybridNoiseReduction as HNR
import imageio


class HNRTest(unittest.TestCase):

	def test_rgb(self):
		rgb_noisy = imageio.imread('imgs/noisy1.jpg')
		hnr = HNR()
		rgb_denoise = hnr.run(rgb_noisy)
