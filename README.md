# DD2424_Project

# Install git lfs
```
git lfs install
git lfs track "*.tar.gz"
git add .gitattributes
```


# Obtaining the Datasets
 Run in terminal:
 ```
 mkdir Data/Data
 cd Data
 tar xvfz annotations.tar.gz -C Data/
 tar xvfz images.tar.gz -C Data/
 ```

# Build splits
Run once from the repo root. This copies ~7350 images (~800 MB) from `data/Data/images/`
into a class-named folder tree at `data/folders/{train,val,test}/<class>/<image>.jpg`,
to use `torchvision.datasets.ImageFolder` to load imgs.
```
python src/data/build_splits.py
```