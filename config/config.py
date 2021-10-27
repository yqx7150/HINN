import argparse

BATCH_SIZE = 2

DATA_PATH = "./data/"



def get_arguments():
    parser = argparse.ArgumentParser(description="training codes")
    
    parser.add_argument("--task", type=str, help="Name of this training")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE, help="Batch size for training. ")       
    parser.add_argument("--phase_weight", type=float, default=10, help="Weight for phase loss. ")                 
    
    
    return parser
