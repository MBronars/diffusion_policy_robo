from diffusion_policy.common.robomimic_util import RobomimicAbsoluteActionConverter

converter = RobomimicAbsoluteActionConverter('/home/MBronars/workspace/datasets/ICRA/2block/2block_image_200*.hdf5')

for i in range(converter.__len__()):
    converter.convert_idx(i)
