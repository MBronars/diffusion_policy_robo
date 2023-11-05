import h5py
import math
import numpy as np
import matplotlib.pyplot as plt
import random
from matplotlib.cm import get_cmap
import sys

def get_legibility(filename = "/srv/rl2-lab/flash8/mbronars3/ICRA/results/ablations/outputs/alpha.1_beta.1_gamma0.01.hdf5", n_action_steps = 4):
    n_action_steps = int(n_action_steps)
    
    fig = plt.figure(figsize=(12,5))

    ax1 = fig.add_subplot(131, projection="3d")
    ax1.view_init(elev=30, azim=40)
    ax1.set_box_aspect([1,1,1])

    ax2 = fig.add_subplot(132, projection="3d")
    ax2.view_init(elev=30, azim=0)
    ax2.set_box_aspect([1,1,1])

    ax3 = fig.add_subplot(133, projection="3d")
    ax3.view_init(elev=30, azim=-40)
    ax3.set_box_aspect([1,1,1])

    #Legend for success plots
    circle1 = plt.Line2D([], [], color='red', linewidth=3)
    circle2 = plt.Line2D([], [], color='green', linewidth=3)
    circle3 = plt.Line2D([], [], color='black', linewidth=3)

    fig.legend([circle1, circle2, circle3], ['Picked Up Red', 'Picked Up Green', 'Failure'], loc='upper center', fontsize = 15, ncol=3)

    total_success = 0
    total_red = 0
    total_green = 0
    total = 0

    x_min = math.inf
    x_max = -math.inf
    y_min = math.inf
    y_max = -math.inf
    z_min = math.inf
    z_max = -math.inf

    max_green_legibility = 0
    min_green_legibility = math.inf
    max_red_legibility = 0
    min_red_legibility = math.inf



    red_legibility_list = []
    green_legibility_list = []
    total_legibility = []

    demo_file = "/srv/rl2-lab/flash8/mbronars3/RAL/datasets/block_reach.hdf5"
    with h5py.File(demo_file, "r") as f:
        demo_lengths = []
        for index, demo in enumerate(f['data']):

    
            # total += 1
            trajectory = f['data'][demo]['obs']['robot0_eef_pos'][()]
            final_heights = f['data'][demo]['obs']['object'][:, [2, 9]][-1]
            final_heights_path = f['data'][demo]['obs']['object'][:, [2, 9]]
            # success = f['data'][demo]['rewards'][()]

            success = final_heights[0] > 0.84 or final_heights[1] > 0.84
            # if not success:
            #     c = 'black'
            # else:
            #     total_success += 1
            #     if final_heights[0] > final_heights[1]:
            #         total_red += 1
            #         c = get_cmap('Reds')(random.randint(50, 90)/100)
            #     else:
            #         total_green += 1
            #         c = get_cmap('Greens')(random.randint(50, 90)/100)

            red_pos = f['data'][demo]["obs"]["object"][:, 0:3] 
            green_pos = f['data'][demo]["obs"]["object"][:, 7:10]

            # only keep every n_action_steps
            trajectory = trajectory[::n_action_steps]
            red_pos = red_pos[::n_action_steps]
            green_pos = green_pos[::n_action_steps]

            # get position where object is picked up
            successes = final_heights_path > 0.84
            # get index of first success
            for i, x in enumerate(successes):
                if x.any():
                    trajectory = trajectory[:i]
                    red_pos = red_pos[:i]
                    green_pos = green_pos[:i]
                    break

            red = False

            scale = range(1, 1 + len(trajectory))
            scale = [x**2 for x in scale]

            if success and final_heights[0] > final_heights[1]:
                legibility = np.linalg.norm(trajectory - green_pos, axis=1)
                legibility = sum(legibility / scale)
                red_legibility_list.append(legibility)

                

                if legibility < min_red_legibility:
                    min_red_legibility = legibility
                if legibility > max_red_legibility:
                    max_red_legibility = legibility
                red = True
            elif success:
                legibility = np.linalg.norm(trajectory - red_pos, axis=1) 
                legibility = sum(legibility / scale)
                green_legibility_list.append(legibility)

                if legibility < min_green_legibility:
                    min_green_legibility = legibility
                if legibility > max_green_legibility:
                    max_green_legibility = legibility
            demo_lengths.append(len(trajectory))
        
        # print the variance in demo lengths
        print(f"Variance in demo lengths: {np.var(demo_lengths)}")



    test_red_legibility_list = []
    test_green_legibility_list = []
    
    with h5py.File(filename, "r") as f:
        test_lengths = []        
        for index, demo in enumerate(f['data']):
            _, length, _, _ = f['data'][demo]['next_obs'].shape
            for i in range(length):
                total += 1
                
                trajectory = f['data'][demo]['next_obs'][:, i, 1, [20, 21, 22]]
                heights = f['data'][demo]['next_obs'][:, i, 1, [2, 9]]

                red_pos = f['data'][demo]['next_obs'][:, i, 1, [0, 1, 2]]
                green_pos = f['data'][demo]['next_obs'][:, i, 1, [7, 8, 9]]

                successes = f['data'][demo]['rewards'][:, i]
                
                # get index of first 1 in successes

                success = successes.any()
                stop_index = np.argmax(successes, axis=0)
                red = heights[stop_index-1, 0] > heights[stop_index-1, 1]

                if success:
                    trajectory = trajectory[:stop_index]
                    red_pos = red_pos[:stop_index]
                    green_pos = green_pos[:stop_index]
                
                # red = successes[:, 0].any()

                # if index%2 == 0:
                #     success = success and red
                # else:
                #     success = success and not red

                # for i, x in enumerate(successes):
                #     if x.any():
                #         trajectory = trajectory[:i]
                #         red_pos = red_pos[:i]
                #         green_pos = green_pos[:i]
                #         break
                
                
                
                scale = range(1, 1 + len(trajectory))
                scale = [x**2 for x in scale]
                if success and red:
                    legibility = np.linalg.norm(trajectory - green_pos, axis=1)
                    legibility = sum(legibility / scale)
                    test_red_legibility_list.append(legibility)

                    legibility = (legibility - min_red_legibility) / (max_red_legibility - min_red_legibility)
                    total_legibility.append(legibility)
                    total_red += 1
                    total_success += 1
                elif success:
                    legibility = np.linalg.norm(trajectory - red_pos, axis=1) 
                    legibility = sum(legibility / scale)
                    legibility = (legibility - min_green_legibility) / (max_green_legibility - min_green_legibility)
                    total_legibility.append(legibility)
                    test_green_legibility_list.append(legibility)
                    total_green += 1
                    total_success += 1
                if not success:
                    c = 'black'
                elif red:
                    c = get_cmap('Reds')(random.randint(50, 90)/100)
                else:
                    c = get_cmap('Greens')(random.randint(50, 90)/100)


                # Extract x, y, and z coordinates from trajectory
                x = trajectory[:, 0]
                y = trajectory[:, 1]
                z = trajectory[:, 2]

                #change max and min values of x, y, and z if necessary
                if x.min() < x_min and x.min() > -.3:
                    x_min = x.min()
                if x.max() > x_max and x.max() < .3:
                    x_max = x.max()
                if y.min() < y_min and y.min() > -.3:
                    y_min = y.min()
                if y.max() > y_max and y.max() < .3:
                    y_max = y.max()
                if z.min() < z_min and z.min() > -1:
                    z_min = z.min()
                if z.max() > z_max and z.max() < 1.1:
                    z_max = z.max()
                
                ax1.plot(x, y, z, color = c)
                ax2.plot(x, y, z, color = c)
                ax3.plot(x, y, z, color = c)

                test_lengths.append(len(trajectory))
        # print the variance in test lengths
        print(f"Variance in test lengths: {np.var(test_lengths)}")

            #set axis limits    
    ax1.set_xlim(x_min, x_max)
    ax1.set_ylim(y_min, y_max)
    ax1.set_zlim(z_min, z_max)

    ax1.set_xticklabels([])
    ax1.set_yticklabels([])
    ax1.set_zticklabels([])

    ax2.set_xlim(x_min, x_max)
    ax2.set_ylim(y_min, y_max)
    ax2.set_zlim(z_min, z_max)

    ax2.set_xticklabels([])
    ax2.set_yticklabels([])
    ax2.set_zticklabels([])

    ax3.set_xlim(x_min, x_max)
    ax3.set_ylim(y_min, y_max)
    ax3.set_zlim(z_min, z_max)

    ax3.set_xticklabels([])
    ax3.set_yticklabels([])
    ax3.set_zticklabels([])

    red_leg = np.array(red_legibility_list)
    green_leg = np.array(green_legibility_list)
    test_red_leg = np.array(test_red_legibility_list)
    test_green_leg = np.array(test_green_legibility_list)

    mean_test_red = np.mean(test_red_leg)
    mean_test_green = np.mean(test_green_leg)

    mean_percentile = np.sum(red_leg < mean_test_red) / len(red_leg)
    mean_percentile += np.sum(green_leg < mean_test_green) / len(green_leg)
    mean_percentile /= 2

    total_percentile = []

    for r in test_red_leg:
        red_percent = np.sum(red_leg < r) / len(red_leg)
        total_percentile.append(red_percent)
    
    for g in test_green_leg:
        green_percent = np.sum(green_leg < g) / len(green_leg)
        total_percentile.append(green_percent)

    tp = np.array(total_percentile)
    print(f"Mean percentile: {mean_percentile}")
    # print std of legibility
    print(f"Std percentile: {np.std(tp)}")


    # from IPython import embed; embed()

    if len(total_legibility) > 0:
        final_legibility = np.mean(total_legibility)
    else:
        final_legibility = 0
    success_rate = total_success / total


    # round(np.mean(tp), 4)) + " +- " + str(round(np.std(tp), 4)
    # Display total success, red, and green
    if total_success > 0:
        fig.text(0.2, 0.05, 'Success Rate: ' + str(round(total_success/total, 4)), ha='center', fontsize=12)
        fig.text(0.4, 0.05, 'Legibility: ' + str(final_legibility), ha='center', fontsize=12)
        # fig.text(0.5, 0.05, 'Red Rate: ' + str(round(total_red/total_success, 2)), ha='center', fontsize=12)
        # fig.text(0.7, 0.05, 'Green Rate: ' + str(round(total_green/total_success, 2)), ha='center', fontsize=12)
        fig.text(0.6, 0.05 , 'Num Green: ' + str(total_green), ha='center', fontsize=12)
        fig.text(0.8, 0.05, 'Num Red: ' + str(total_red), ha='center', fontsize=12)


    #image_root = "/srv/rl2-lab/flash8/mbronars3/legibility/full_training_scan/unet/abs/images/"
    img_name = filename.split("/")[-1]
    image_root = filename.split("/")[:-1]
    image_root = "/".join(image_root) + "/" #images/"

    img_name = img_name[:-5]
    plt.savefig(image_root + img_name + ".png")

    print((final_legibility, success_rate))
    print(image_root + img_name + ".png")
    # print(f"Max green legibility: {max_green_legibility}")
    # print(f"Min green legibility: {min_green_legibility}")
    # print(f"Max red legibility: {max_red_legibility}")
    # print(f"Min red legibility: {min_red_legibility}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        # from IPython import embed; embed()
        print("Usage: python evaluate_legibility.py file_path.hdf5> n_action_steps")
        get_legibility()
    else:
        filename = sys.argv[1]
        n_action_steps = int(sys.argv[2])
        get_legibility(filename, n_action_steps=n_action_steps)


    


# Copy this snippet back into the with statement to compute min and max legibility
# max_green_legibility = 0
# max_red_legibility = 0

# min_green_legibility = math.inf
# min_red_legibility = math.inf
# for demo in f['data']:
#     total += 1
#     trajectory = f['data'][demo]['obs']['robot0_eef_pos'][()]
#     final_heights = f['data'][demo]['obs']['object'][:, [2, 9]][-1]
#     # success = f['data'][demo]['rewards'][()]

#     success = final_heights[0] > 0.86 or final_heights[1] > 0.86
#     if not success:
#         c = 'black'
#     else:
#         total_success += 1
#         if final_heights[0] > final_heights[1]:
#             total_red += 1
#             c = get_cmap('Reds')(random.randint(50, 90)/100)
#         else:
#             total_green += 1
#             c = get_cmap('Greens')(random.randint(50, 90)/100)

#     red_pos = f['data'][demo]["obs"]["object"][:, 0:3] 
#     green_pos = f['data'][demo]["obs"]["object"][:, 7:10]
#     red = False

#     if success and final_heights[0] > final_heights[1]:
#         legibility = np.linalg.norm(trajectory - green_pos, axis=1)
#         legibility = sum(legibility / range(1, 1+ len(legibility)))
#         if legibility < min_red_legibility:
#             min_red_legibility = legibility
#         if legibility > max_red_legibility:
#             max_red_legibility = legibility
#         red = True
#     elif success:
#         legibility = np.linalg.norm(trajectory - red_pos, axis=1) 
#         legibility = sum(legibility / range(1, 1 + len(legibility)))
#         if legibility < min_green_legibility:
#             min_green_legibility = legibility
#         if legibility > max_green_legibility:
#             max_green_legibility = legibility