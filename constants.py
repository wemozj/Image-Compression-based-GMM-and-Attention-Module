import os



# other codec tested data saved in OTHER_CODES_ROOT
OTHER_CODECS_ROOT = os.environ.get('OTHER_CODECS_ROOT', '/home/zhujun/IMAGE_COMPRESSION/IC2020/data/unit_test_output')

# VALIDATION_DATASETS_ROOT = os.environ.get('VAL_ROOT', '/data/zhujun/IC_images/test')

# CONFIG_BASE = os.environ.get('CONFIG_BASE', '/home/zhujun/IMAGE_COMPRESSION/last_ic_repo/configs/base_config.json')
#
# CONFIG_BASE_AE = os.environ.get('CONFIG_BASE_AE', 'ae_configs')
# CONFIG_BASE_PC = os.environ.get('CONFIG_BASE_PC', 'pc_configs')
#
# NUM_PREPROCESS_THREADS = int(os.environ.get('NUM_PREPROCESS_THREADS', 4))
# NUM_CROPS_PER_IMG = int(os.environ.get('NUM_CROPS_PER_IMG', 1))

# Transcribed from paper
# bpp, ms-ssim
_RIPPEL_KODAK = [
    (.095, .92),
    (.14,  .94),
    (.2,   .956),
    (.3,   .97),
    (.4,   .9783),
    (.5,   .983),
    (.6,   .9858),
    (.7,   .9880),
    (.8,   .9897),
    (.9,   .9914),
    (1.0,  .9923),
    (1.1,  .9935),
    (1.2,  .994),
    (1.3,  .9946),
    (1.4,  .9954)
]

#Balle
#(bpp, ms-ssim, ssim, pnsr)
_Balle_KODAK = [
    (0.0869, 0.9223, 0.7306, 24.7876),
    (0.1464, 0.9545, 0.7997, 26.4126),
    (0.2354, 0.9739, 0.8577, 27.8904),
    (0.3471, 0.9828, 0.8852, 28.8467),
    (0.5444, 0.9902, 0.9341, 39.9030),
    (0.8070, 0.9942, 0.9341, 30.9030),
    (1.2192, 0.9966, 0.9732, 34.2845),
    (1.8324, 0.9980, 0.9840, 35.5577)
]

#Balle
#(bpp, ms-ssim, ssim, pnsr)
_Own_KODAK = [
    (0.0869, 0.9253, 0.7306, 24.7876),
    (0.1464, 0.9595, 0.7997, 26.4126),
    (0.2354, 0.9799, 0.8577, 27.8904),
    (0.3471, 0.9900, 0.8852, 28.8467),
    (0.5444, 0.9950, 0.9341, 39.9030),
    (0.8070, 0.9930, 0.9341, 30.9030),
    (1.2192, 0.9987, 0.9732, 34.2845),
    (1.8324, 0.9989, 0.9840, 35.5577)
]