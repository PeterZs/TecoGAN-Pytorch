import argparse
import os.path
import glob

import imageio
from skimage import img_as_float, img_as_ubyte

from hnr import HybridNoiseReduction


def run_denoise(input_fn):
	input_fn_base = os.path.splitext(os.path.basename(input_fn))[0]

	rgb_noisy = imageio.imread(input_fn)
	hnr = HybridNoiseReduction()
	rgb_denoise, mask = hnr.run(img_as_float(rgb_noisy),
								nr_method=args.nr_method,
								nr_parm=args.nr_parm,
								no_blend=args.no_blend,
								mask_level=args.mask_level,
								blend_morph_kernel=args.blend_morph_kernel)

	# mask_fn = 'outputs/{}_mask_{}_parm_{}_level_{}_morph_{}.png'.format(input_fn_base, args.nr_method,
	# 																	str(args.nr_parm),
	# 																	str(args.mask_level),
	# 																	str(args.blend_morph_kernel))
	# out_fn = 'outputs/{}_output_{}_parm_{}_level_{}_morph_{}.jpg'.format(input_fn_base, args.nr_method,
	# 																	 str(args.nr_parm),
	# 																	 str(args.mask_level),
	# 																	 str(args.blend_morph_kernel))
	out_fn = 'outputs/{}.jpg'.format(input_fn_base)
	# imageio.imwrite(mask_fn, img_as_ubyte(mask))
	imageio.imwrite(out_fn, img_as_ubyte(rgb_denoise))

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--input_dir', required=True, type=str, help='input directory')
parser.add_argument('--nr_method', default='freq', type=str, help='denoising method')
parser.add_argument('--nr_parm', default=0.05, type=float, help='denoising parm')
parser.add_argument('-no_blend', action='store_true', help='do not apply blending mask')
parser.add_argument('--mask_level', default=-2, type=int, help='laplacian pyramid blending level')
parser.add_argument('--blend_morph_kernel', default=1, type=int,
					help='kernel size for morphlogical operation default 1 is none')
args = parser.parse_args()

filenames = sorted(glob.glob(args.input_dir))

for i, file in enumerate(filenames):
	print('Processing image {}th image_{}'.format(str(i), file))
	run_denoise(file)

