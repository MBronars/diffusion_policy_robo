from diffusion_policy.common.robomimic_util import RobomimicAbsoluteActionConverter

converter = RobomimicAbsoluteActionConverter('/srv/rl2-lab/flash8/mbronars3/ICRA/datasets/2block_random_abs.hdf5')

for i in range(converter.__len__()):
    converter.convert_idx(i)
