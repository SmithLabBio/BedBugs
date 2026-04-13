import pandas as pd
from sklearn.model_selection import train_test_split
import os
import argparse
import numpy as np

def split_subjects(metadata, test_size, val_size, seed):
    hem = metadata[metadata['species'] == 'hemipterus']['subject'].unique()
    lec = metadata[metadata['species'] == 'lectularius']['subject'].unique()

    labels = ['hemipterus'] * len(hem) + ['lectularius'] * len(lec)
    subjects = hem.tolist() + lec.tolist()

    X_train, X_temp, y_train, y_temp = train_test_split(
        subjects, labels, test_size=test_size+val_size,
        stratify=labels, random_state=seed
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        test_size=test_size/(test_size+val_size),
        stratify=y_temp, random_state=seed
    )

    return X_train, X_val, X_test


def build_df(subjects, metadata):
    return metadata[metadata['subject'].isin(subjects)].copy()


def prepare_splits(metadata_path, outdir, test_size, val_size, seed, im_path):
    os.makedirs(outdir, exist_ok=False)

    metadata = pd.read_csv(metadata_path)

    metadata["spName"] = np.where(
        metadata["species"] == "hemipterus",
        "Cimex_hemipterus",
        "Cimex_lectularius"
    )
    metadata["Path"] = metadata.apply(
        lambda row: os.path.join(im_path, row["spName"], row["Filename"]),
        axis=1
    )
    sophie = metadata[metadata.source == "Sophie"]
    inat   = metadata[metadata.source == "iNaturalist"]

    s_tr, s_val, s_te = split_subjects(sophie, test_size, val_size, seed)
    i_tr, i_val, i_te = split_subjects(inat, test_size, val_size, seed)

    splits = {
        "sophie_train": build_df(s_tr, sophie),
        "sophie_val":   build_df(s_val, sophie),
        "sophie_test":  build_df(s_te, sophie),
        "inat_train":   build_df(i_tr, inat),
        "inat_val":     build_df(i_val, inat),
        "inat_test":    build_df(i_te, inat),
    }

    # Combined training
    splits["combined_train"] = pd.concat(
        [splits["sophie_train"], splits["inat_train"]]
    )

    for name, df in splits.items():
        df.to_csv(os.path.join(outdir, f"{name}.csv"), index=False)

def parse_arguments():
    parser = argparse.ArgumentParser(description="Prepare train/val/test splits for Bed Bug CNN.")
    parser.add_argument("-m", "--metadata_path", type=str, required=True, help="Path to the combined metadata CSV file.")
    parser.add_argument("-o", "--outdir", type=str, required=True, help="Directory to save the split metadata files.")
    parser.add_argument("--test_size", type=float, default=0.2, help="Proportion of data to use as test set.")
    parser.add_argument("--val_size", type=float, default=0.2, help="Proportion of data to use as validation set.")
    parser.add_argument("--random_state", type=int, default=1234, help="Random seed for reproducibility.")
    parser.add_argument("--im_path", type=str, help="Base path to images.")
    return parser.parse_args()

def main():
    args = parse_arguments()
    prepare_splits(args.metadata_path, args.outdir, args.test_size, args.val_size, args.random_state, args.im_path)

if __name__ == "__main__":
    main()