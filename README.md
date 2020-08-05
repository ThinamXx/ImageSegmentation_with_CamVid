# **Image Segmentation with CamVideo**

**Fastai Library or API**
- [Fast.ai](https://www.fast.ai/about/) is the first deep learning library to provide a single consistent interface to all the most commonly used deep learning applications for vision, text, tabular data, time series, and collaborative filtering.
- [Fast.ai](https://www.fast.ai/about/) is a deep learning library which provides practitioners with high-level components that can quickly and easily provide state-of-the-art results in standard deep learning domains, and provides researchers with low-level components that can be mixed and matched to build new approaches.

**Image Segmentation**
- In digital image processing and computer vision, image segmentation is the process of partitioning a digital image into multiple segments. The goal of segmentation is to simplify and/or change the representation of an image into something that is more meaningful and easier to analyze.

**Preparing the Model**
- I have used [Fastai](https://www.fast.ai/about/) API to train the Model. It seems quite challenging to understand the code if you have never encountered with Fast.ai API before.
One important note for anyone who has never used Fastai API before is to go through [Fastai Documentation](https://docs.fast.ai/). And if you are using Fastai in Jupyter Notebook then you can use doc(function_name) to get the documentation instantly.

**Dataset**
- Fastai has its own [Dataset](https://docs.fast.ai/datasets.html).I have used [Fastai CAMVID Dataset](https://course.fast.ai/datasets) using the following lines of codes:

```javascript
untar_data(URLs.CAMVID)
```

**Image Segmentation with Fastai**
- In digital image processing and computer vision, image segmentation is the process of partitioning a digital image into multiple segments. The goal of segmentation is to simplify and/or change the representation of an image into something that is more meaningful and easier to analyze.

- **Creating Segmentation ItemList**

```javascript
(SegmentationItemList.from_folder(path_img)
      .split_by_fname_file('../valid.txt')
      .label_from_func(get_y_fn, classes=codes))
```

- **Creating Data for Segmentation**
  
 ```javascript
  (src.transform(get_transforms(), size=size, tfm_y=True)
        .databunch(bs=bs)
        .normalize(imagenet_stats))
 ```

**Creating Model with Fastai API**

```javascript
unet_learner(data, models.resnet34, metrics=metrics, wd=wd)
```

**Image Segmentation**

![Image](https://res.cloudinary.com/dge89aqpc/image/upload/v1596630592/Seg_spjwsr.png)

**Accuracy of the Model**
- Fastai API is so powerful and it gives good accuracy result. Snapshot of the Accuracy is shown below:

![Image](https://res.cloudinary.com/dge89aqpc/image/upload/v1596630819/Acc_oyvfg4.png)

**Snapshot of the Loss Function**
- Loss Function is gradually decreasing so the Model is not Overfitting.

![Image](https://res.cloudinary.com/dge89aqpc/image/upload/v1596630918/Leaern_abn9x8.png)

**Snapshot of the Learning Rate**

![Image](https://res.cloudinary.com/dge89aqpc/image/upload/v1596631100/Learnin_it1vrq.png)
 
