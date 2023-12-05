import eval
import argparse
import click

guidance_weights = [1, 2, 5, 10, 15, 20]
alphas = [0]
gammas = [1]
# alphas = [0, 0.4, 0.6, 0.8, 1]
# gammas = [0.2, 0.4, 0.7, 0.9, 1]
# guidance_weights = [15]
# alphas = [.9]
# gammas = [0.4]
# seeds = [0]
# seeds = [0]

def run_ablations(directory, checkpoint, alphas = None, gammas = None, guidance_weights = None):
    tuned_alpha = 0.9
    tuned_gamma = 0.4
    tuned_guidance_weight = 10
    for alpha in alphas:
        for guidance_weight in guidance_weights:
            for gamma in gammas:
                # for seed in seeds:
                with click.Context(eval.main) as ctx:
                    with ctx.scope(cleanup=False):
                        output_dir = f"{directory}/alpha/{alpha}/w_{guidance_weight}/gamma/{gamma}"
                        try:
                            eval.main(['-c', checkpoint, '-o', output_dir, '-a', str(alpha), '-g', str(gamma), '-w', str(guidance_weight)])
                        except SystemExit:
                            pass
                        except KeyboardInterrupt:
                            print("Interrupted by user")
                            raise

    # if alphas is not None:
    #     for alpha in alphas:
    #         for guidance_weight in guidance_weights:
    #             with click.Context(eval.main) as ctx:
    #                 with ctx.scope(cleanup=False):
    #                     output_dir = f"{directory}/alpha/{alpha}/w_{guidance_weight}"
    #                     try:
    #                         eval.main(['-c', checkpoint, '-o', output_dir, '-a', str(alpha), '-g', str(tuned_gamma), '-w', str(guidance_weight)])
    #                     except SystemExit:
    #                         pass
    #                     except KeyboardInterrupt:
    #                         print("Interrupted by user")
    #                         raise
    # if gammas is not None:
    #     for gamma in gammas:
    #         for guidance_weight in guidance_weights:
    #             with click.Context(eval.main) as ctx:
    #                 with ctx.scope(cleanup=False):
    #                     output_dir = f"{directory}/gamma/{gamma}/w_{guidance_weight}"
    #                     try:
    #                         eval.main(['-c', checkpoint, '-o', output_dir, '-a', str(tuned_alpha), '-g', str(gamma), '-w', str(guidance_weight)])
    #                     except SystemExit:
    #                         pass
    #                     except KeyboardInterrupt:
    #                         print("Interrupted by user")
    #                         raise
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--directory', type=str, default='ablations')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/robo_diffusion_policy.pt')
    args = parser.parse_args()

    for alpha in alphas:
        for gamma in gammas:
            run_ablations(args.directory, args.checkpoint, [alpha], [gamma], guidance_weights)
    # run_ablations(args.directory, args.checkpoint, alphas, gammas, guidance_weights)

if __name__ == '__main__':
    main()

