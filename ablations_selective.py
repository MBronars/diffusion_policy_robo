import eval_selective as eval
import argparse
import click

# guidance_weights = [0.5, 1, 5, 10, 15, 20]
# alphas = [0, 0.2, 0.4, 0.6, 0.8, 1]
# gammas = [0.5, 0.6, 0.7, 0.8, 0.9, 1]
guidance_weights = [0, 1, 2, 5, 10]
alphas = [0, .4, .6, .8, 0.9, 1]
gammas = [1, 0.95, 0.75, 0.5, 0.25]
# seeds = [420]

tuned_alpha = 0.9
tuned_gamma = 1

def run_ablations(directory, checkpoint, alphas = None, gammas = None, guidance_weights = None):
    
    # for alpha in alphas:
        # for guidance_weight in guidance_weights:
        #     for gamma in gammas:
        #         for seed in seeds:
        #             with click.Context(eval.main) as ctx:
        #                 with ctx.scope(cleanup=False):
        #                     output_dir = f"{directory}/alpha/{alpha}/w_{guidance_weight}/gamma/{gamma}/seed/{seed}"
        #                     try:
        #                         eval.main(['-c', checkpoint, '-o', output_dir, '-a', alpha, '-g', gamma, '-w', guidance_weight, '-s', seed,'-t', ["mkbtsh", "mkbtlh"]])
        #                     except SystemExit:
        #                         pass
        #                     except KeyboardInterrupt:
        #                         print("Interrupted by user")
        #                         raise
    if alphas is not None:
        for alpha in alphas:
            for guidance_weight in guidance_weights:
                with click.Context(eval.main) as ctx:
                    with ctx.scope(cleanup=False):
                        output_dir = f"{directory}/alpha/{alpha}/w_{guidance_weight}"
                        try:
                            eval.main(['-c', checkpoint, '-o', output_dir, '-a', alpha, '-g', tuned_gamma, '-w', guidance_weight,'-t', ["mkbts", "mkbtl"]])
                        except SystemExit:
                            pass
                        except KeyboardInterrupt:
                            print("Interrupted by user")
                            raise
    if gammas is not None:
        for gamma in gammas:
            for guidance_weight in guidance_weights:
                with click.Context(eval.main) as ctx:
                    with ctx.scope(cleanup=False):
                        output_dir = f"{directory}/gamma/{gamma}/w_{guidance_weight}"
                        try:
                            eval.main(['-c', checkpoint, '-o', output_dir, '-a', tuned_alpha, '-g', gamma, '-w', guidance_weight,'-t', ["mkbts", "mkbtl"]])
                        except SystemExit:
                            pass
                        except KeyboardInterrupt:
                            print("Interrupted by user")
                            raise
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--directory', type=str, default='ablations')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/robo_diffusion_policy.pt')
    args = parser.parse_args()

    # for alpha in alphas:
    #     for gamma in gammas:
    #     run_ablations(args.directory, args.checkpoint, [alpha], [gamma], guidance_weights)
    run_ablations(args.directory, args.checkpoint, alphas, gammas, guidance_weights)

if __name__ == '__main__':
    main()

