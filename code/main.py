
# TODO: make sure these are actually used here, otherwise get rid of them 
import argparse # will definitely need to parse arguments 
import torch # pytorch stuff 


# main entrypoint of application 
if (__name__ == "__main__"):
    print(f'CUDA Availability: {torch.cuda.is_available()}')
    # TODO: Fill this out properly 