# ExDark Dataset Setup & Annotation Conversion

## 1. Download the ExDark Dataset

- **Images**: [Download from Google Drive](https://drive.google.com/file/d/1BHmPgu8EsHoFDDkMGLVoXIlCth2dW6Yx/view)
- **Annotations**: [Download from Google Drive](https://drive.google.com/file/d/1P3iO3UYn7KoBi5jiUkogJq96N6maZS1i/view)
- **Class List**: [Download imageclasslist.txt](https://github.com/cs-chan/Exclusively-Dark-Image-Dataset/blob/master/Groundtruth/imageclasslist.txt)

After downloading and extracting, organize the files as follows:

data/ExDark/
├── images/
├── annotations/
└── imageclasslist.txt

yaml
Copy
Edit

---

## 2. Convert Annotations to COCO Format

Run the following script to generate a COCO-style ground truth JSON file:

```bash
python src/preprocessing/exdark_to_coco.py \
  --exdark_root data/ExDark \
  --output data/ExDark/ground_truth.json
```
