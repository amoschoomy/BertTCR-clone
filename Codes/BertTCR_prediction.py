import torch
import torch.nn as nn
import argparse
import os
import numpy as np
import math
import matplotlib.pyplot as plt
import torch.utils.data as Data


class BertTCR(nn.Module):
    def __init__(self, filter_num, kernel_size, ins_num, drop_out):
        super(BertTCR, self).__init__()
        print(f"Initializing BertTCR with parameters:")
        print(f"filter_num: {filter_num}")
        print(f"kernel_size: {kernel_size}")
        print(f"ins_num: {ins_num}")
        print(f"drop_out: {drop_out}")
        
        self.filter_num = filter_num
        self.kernel_size = kernel_size
        self.ins_num = ins_num
        self.drop_out = drop_out
        
        self.convs = nn.ModuleList([
            nn.Sequential(nn.Conv1d(in_channels=768,
                                    out_channels=filter_num[idx],
                                    kernel_size=h,
                                    stride=1),
                          nn.Sigmoid(),
                          nn.AdaptiveMaxPool1d(1))
            for idx, h in enumerate(kernel_size)
        ])
        
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(sum(filter_num), 1)
        self.models = nn.ModuleList([nn.Linear(ins_num, 2) for _ in range(5)])
        self.dropout = nn.Dropout(p=drop_out)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        print(f"Original input shape: {x.shape}")
        
        # Handle padding if sequence length is less than expected
        if x.shape[1] < 24:
            padding_size = 24 - x.shape[1]
            x = torch.nn.functional.pad(x, (0, 0, 0, padding_size, 0, 0))
            print(f"Shape after padding: {x.shape}")
        
        batch_size = x.shape[0]
        x = x.permute(0, 2, 1)
        print(f"Shape after permute: {x.shape}")
        
        # Debug convolution operations
        out = [conv(x) for conv in self.convs]
        print(f"Shapes after individual convolutions: {[o.shape for o in out]}")
        
        if not out:  # Check if out is empty
            print("Error: Convolution output is empty. Skipping computation.")
            return None

        out = torch.cat(out, dim=1)
        print(f"Shape after concatenation: {out.shape}")

        out = out.reshape(-1, 1, sum(self.filter_num))
        print(f"Shape after first reshape: {out.shape}")

        out = self.dropout(self.fc(out))
        print(f"Shape after FC layer and dropout: {out.shape}")

        if out.numel() == 0:
            print("Error: Output tensor has no elements. Skipping prediction.")
            return None

        try:
            out_squeezed = out.squeeze()
            print(f"Shape before final padding: {out_squeezed.shape}")
            
            # Only pad if out_squeezed is valid
            if len(out_squeezed.shape) > 0:
                out = torch.nn.functional.pad(out_squeezed, (0, self.ins_num - out_squeezed.shape[0])).unsqueeze(0)
            else:
                print("Warning: Invalid shape, skipping padding.")
                return None
            
            print(f"Shape after final reshape: {out.shape}")
        
        except Exception as e:
            print(f"Error during reshaping: {e}")
            return None

        pred_sum = 0
        for i, model in enumerate(self.models):
            pred = self.dropout(model(out))
            print(f"Shape after model {i}: {pred.shape}")
            pred_sum += pred

        out = self.sigmoid(pred_sum / len(self.models))
        print(f"Final output shape: {out.shape}")

        return out


def create_parser():
    parser = argparse.ArgumentParser(
        description="Script to predict samples using BertTCR.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--sample_dir",
        dest="sample_dir",
        type=str,
        help="The directory of samples for prediction.",
        default="/scratch/project/tcr_ml/BertTCR/sarcoma_embedding",
    )
    parser.add_argument(
        "--model_file",
        dest="model_file",
        type=str,
        help="The pretrained model file for prediction in .pth format.",
        default="./Model/Pretrained_Lung.pth",
    )
    parser.add_argument(
        "--tcr_num",
        dest="tcr_num",
        type=int,
        help="The number of TCRs in each sample.",
        default=100,
    )
    parser.add_argument(
        "--max_length",
        dest="max_length",
        type=int,
        help="The maximum of TCR length.",
        default=24,
    )
    parser.add_argument(
        "--kernel_size",
        dest="kernel_size",
        type=list,
        help="The size of kernels in the convolutional layer.",
        default=[2,3,4],
    )
    parser.add_argument(
        "--filter_num",
        dest="filter_num",
        type=list,
        help="The number of the filters with corresponding kernel sizes.",
        default=[3,2,1],
    )
    parser.add_argument(
        "--dropout",
        dest="dropout",
        type=float,
        help="The dropout rate in one-layer linear classifiers.",
        default=0.4,
    )
    parser.add_argument(
        "--device",
        dest="device",
        type=str,
        help="The device used to make prediction.",
        default="cuda:0",
    )
    parser.add_argument(
        "--output",
        dest="output",
        type=str,
        help="Output file in .tsv format.",
        default='./ZERO_prediction.tsv',
    )
    return parser.parse_args()


if __name__ == "__main__":
    print("\nStarting BertTCR prediction script...")
    
    # Parse arguments
    args = create_parser()
    print("\nArguments:")
    for arg in vars(args):
        print(f"{arg}: {getattr(args, arg)}")
    
    # Load model
    print("\nLoading model...")
    model = BertTCR(filter_num=args.filter_num,
                     kernel_size=args.kernel_size,
                     ins_num=args.tcr_num,
                     drop_out=args.dropout).to(torch.device(args.device))

    model.load_state_dict(torch.load(args.model_file))
    model = model.eval()
    print("Model loaded successfully")

    # Predict samples
    print("\nStarting prediction...")
    sample_dir = args.sample_dir if args.sample_dir[-1] == "/" else args.sample_dir + "/"
    
    with open(args.output, "w", encoding="utf8") as output_file:
        output_file.write("Sample\tProbability\tPrediction\n")
        
        for sample_file in os.listdir(sample_dir):
            print(f"\nProcessing sample: {sample_file}")
            
            # Load sample
            sample_path = os.path.join(sample_dir, sample_file)
            sample = torch.load(sample_path)
            print(f"Loaded sample shape: {sample.shape}")
            print(f"Sample dtype: {sample.dtype}")
            print(f"Sample device: {sample.device}")
            
            # Move to device
            sample = sample.to(torch.device(args.device))
            print(f"Sample moved to device: {sample.device}")
            
            # Generate input
            input_matrix = sample
            print(f"Input matrix shape: {input_matrix.shape}")
            print(f"Input matrix size: {input_matrix.numel()}")
            
            # Make prediction
            print("\nMaking prediction...")
            try:
                predict = model(input_matrix)
                if predict is None:
                    print(f"Skipping sample {sample_file} due to invalid model output.")
                    continue  # Skip this sample
                
                print(f"Prediction shape: {predict.shape}")

                prob = float(1 / (1 + math.exp(-predict[0][1] + predict[0][0])))
                pred = True if prob > 0.5 else False
                
                print(f"Probability: {prob}")
                print(f"Prediction: {pred}")

                # Save result
                output_file.write(f"{sample_file}\t{prob}\t{pred}\n")

            except Exception as e:
                print(f"Error during prediction: {str(e)}")


    print("\nPrediction completed")
    print("The prediction results have been saved to: " + args.output)