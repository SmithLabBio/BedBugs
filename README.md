## Code repository for ML to identify bedbugs from images

# First, process our data, creating training, test, and validation datasets
# NOTE: JUST RUN AS AN INTERACTIVE JOB BEFORE RUNNING EVERYTHING ELSE, use the code below
srun -p normal --pty bash
python BedBugs/preprocess_data.py -m all_metadata.csv -o split_dir --im_path /mnt/scratch/smithlab/megan/maedo_testing/photos

# NOTE: Also before you start, run this: 
mkdir logs 

# NOTE: Before submitting slurm scripts, activate the appropriate conda environment

# NOTE: CREATE SLURM SCRIPTS FOR THE BELOW.
# Train on Sophie data only (DO WITH AND WITHOUT --balance and --augment--so four total runs, change save path for each)
python BedBugs/train_basic_models.py --train_csv split_dir/sophie_train.csv --val_csvs sophie=split_dir/sophie_val.csv --save_path "sophie_training.keras" --balance --augment

# Then, train on combined data (DO WITH AND WITHOUT --balance and --augment--so four total runs, change save path for each)
python BedBugs/train_basic_models.py --train_csv split_dir/combined_train.csv --val_csvs sophie=split_dir/sophie_val.csv inat=split_dir/inat_val.csv --save_path "combined_training.keras" --balance --augment

# Then, train on inat data, finetuning on sophie's data. (DO WITH AND WITHOUT --balance and --augment--so four total runs, change save path for each)
python BedBugs/train_iterative_models.py --train_csv split_dir/inat_train.csv --val_csvs sophie=split_dir/sophie_val.csv inat=split_dir/inat_val.csv --finetune_csv split_dir/sophie_train.csv --save_path "finetune_training.keras" --balance --augment

# Try with pretrained network (DO WITH AND WITHOUT --balance and --augment--so four total runs, change save path for each)
python BedBugs/train_pretrained_models.py --train_csv split_dir/sophie_train.csv --val_csvs sophie=split_dir/sophie_val.csv --save_path "pretrained_training.keras" 