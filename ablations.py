import eval
import argparse

guidance_weights = [1, 5, 10, 15, 20]
alphas = [0, 0.25, 0.5, 0.75, 0.85, 1]
gammas = [0.5, 0.65, 0.75, 0.85, 0.95, 1]

def run_ablations(directory, checkpoint, alphas = None, gammas = None, guidance_weights = None):
    tuned_alpha = 0.85
    tuned_gamma = 0.75
    tuned_guidance_weight = 10
    if alphas is not None:
        for alpha in alphas:
            for guidance_weight in guidance_weights:
                output_dir = f"{directory}/alpha/{alpha}/w_{guidance_weight}"
                eval.main(['-c', checkpoint, '-o', output_dir, '-a', str(alpha), '-g', str(tuned_gamma), '-w', str(guidance_weight)])
    if gammas is not None:
        for gamma in gammas:
            for guidance_weight in guidance_weights:
                output_dir = f"{directory}/gamma/{gamma}/w_{guidance_weight}"
                eval.main(['-c', checkpoint, '-o', output_dir, '-a', str(tuned_alpha), '-g', str(gamma), '-w', str(guidance_weight)])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--directory', type=str, default='ablations')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/robo_diffusion_policy.pt')
    args = parser.parse_args()
    run_ablations(args.directory, args.checkpoint, alphas, gammas, guidance_weights)

if __name__ == '__main__':
    main()

