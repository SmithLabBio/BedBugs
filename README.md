## Code repository for ML to identify bedbugs from images

# First, process our data, creating training, test, and validation datasets
python BedBugs/preprocess_data.py -m all_metadata.csv -o split_dir --im_path /mnt/scratch/smithlab/megan/maedo_testing/photos

# Then, train on combined data
python BedBugs/train_basic_models.py --train_csv split_dir/combined_train.csv --val_csvs sophie=split_dir/sophie_val.csv inat=split_dir/inat_val.csv 

# Then, train on inat data, finetuning on sophie's data.
python BedBugs/train_iterative_models.py --train_csv split_dir/combined_train.csv --val_csvs sophie=split_dir/sophie_val.csv inat=split_dir/inat_val.csv --finetune_csv split_dir/sophie_train.csv 