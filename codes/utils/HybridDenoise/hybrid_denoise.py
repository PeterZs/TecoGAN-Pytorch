import sys
import os
from skimage import img_as_float, img_as_ubyte
sys.path.insert(0, os.path.join(os.getcwd(), "utils/HybridDenoise"))
from ..HybridDenoise.hnr import HybridNoiseReduction

hnr = HybridNoiseReduction()
def hybrid_denoise(rgb_noisy, 
        nr_method="median", 
        nr_parm=7,
        no_blend=False,
        mask_level=-2,
        blend_morph_kernel=1, 
        blend_alpha=0.5,
        ):
    
    rgb_denoise, mask = hnr.run(img_as_float(rgb_noisy),
                                nr_method=nr_method,
                                nr_parm=nr_parm,
                                no_blend=no_blend,
                                mask_level=mask_level,
                                blend_morph_kernel=blend_morph_kernel, 
                                blend_alpha=blend_alpha,
                                )
    
    return img_as_ubyte(rgb_denoise)


if __name__ == "__main__":
    import imageio
    import cv2
    rgb_noisy = imageio.imread("/data/Test_Results/Face_Restoration/internal_test/night_samples/001_001.jpg")
    output_img = hybrid_denoise(rgb_noisy, 
                    nr_method="median", 
                    nr_parm=7,
                    no_blend=False,
                    mask_level=-2,
                    blend_morph_kernel=1, 
            )
    imageio.imwrite("./test_img.jpg", output_img)
