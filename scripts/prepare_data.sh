mkdir -p data
cd data

### npz file
gdown "https://drive.google.com/uc?export=download&id=1HByAk-Hene-UOUzC9y-XJ55t6i4svnWy"
upzip dataset_extras.zip
rm dataset_extras.zip

### VIBE data
gdown "https://drive.google.com/uc?id=1untXhYOLQtpNEy4GTY_0fL_H-k6cTf_r"
unzip vibe_data.zip
rm vibe_data.zip

### Other data
gdown "https://drive.google.com/uc?id=1z3YftRaQvEmxb_f-q_Y63CEDwVCjBxHu"
gdown "https://drive.google.com/uc?id=1DdhMs5PFZMY74-7cffhmBJI5wzyTdnGA"