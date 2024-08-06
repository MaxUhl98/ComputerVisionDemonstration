# Computer Vision Pipeline

Since I heard the [Podcast](https://www.youtube.com/watch?v=RF4LwRl0npQ&t=747s) with Christof Henkel (the guy who was
the #1 competetive kaggler at that time) it became clear to me I want to develop my own ComputerVision model trianing
framework to automate most of the repetetive steps of model training and thus remove the need to create redundant and
poorly documented/tracked experiments in notebooks.
This readme serves as a guide to explain how to use this framework, explain its inner workings and showcase its results
on a real world eye disease dataset.

| Model                                     | Pretrained         |
|-------------------------------------------|--------------------|
| [Efficientnet V2](#Efficientnet V2)       | :heavy_check_mark: |
| [ConvNeXT V2](#ConvNeXT V2)               | :heavy_check_mark: |
| [Vision Transformer](#Vision Transformer) | :heavy_check_mark: |
| [Mini VGG](#Mini VGG)                     | :x:                |

## Index

[Training Workflow](#Training-Workflow)<br>
[Demonstration Workflow](#Demonstration-Workflow)<br>
[Eye Disease Example](#Eye-Disease-Example)<br>
[Models](#Models)<br>
[User Guide](#User-Guide)<br>

## Training Workflow

<img alt="Error" src="demonstration_results/workflow/train_workflow.png?"/>

## Demonstration Workflow

<img alt="Error" src="demonstration_results/workflow/demo_workflow.png?"/>

## Eye Disease Example

Dataset: [Eye Disease Detection Dataset](https://www.kaggle.com/datasets/ysnreddy/eye-disease-detection-dataset) <br>
Original Description by [Surya](https://www.kaggle.com/ysnreddy):<br>

Imagine, for a moment, a world where the beauty of a sunset, the colors of a rainbow, or the
joyful eyes of a loved one start to fade away, not due to the passage of time but because of
a lurking ocular disease. The reality is that millions across the globe face this daunting
experience daily, with conditions like CNV, DME, and Drusen leading the charge. But here's
where you, armed with the prowess of machine learning and a meticulously curated dataset,
can make an indelible mark and shine a beacon of hope in this looming darkness.<br>
Understanding the Adversaries:<br>
● CNV (Choroidal Neovascularization): An eye condition where abnormal blood
vessels grow underneath the retina. These vessels can leak blood and fluid, leading
to a bulge or bump in the macula.<br>
● DME (Diabetic Macular Edema): A consequence of diabetes, this condition is
characterized by fluid accumulation in the macula due to leaking blood vessels.
Without treatment, DME can lead to blindness.<br>
● Drusen: Tiny yellow or white deposits under the retina. They're common as we age,
but a large number in one place or their rapid increase can indicate a problem, such
as age-related macular degeneration (AMD).<br>
And then, of course, we have the eyes that are untouched by these conditions, categorized
under 'Normal'.<br>

Explanation:<br>

The Dataset Images were generated via a common eye disease diagnosing technique called Optical Coherence Tomographie (OTC). The main advantage of this technique is that it is non invasive and thereby does not cause any discomfort in the patient. The goal of this example is to use the training data to train a model that generalizes well on the unseen test data we will use to measure the models performance. 

Example OTC Scan: <br>
![Error](classification/data/Eye_Disease_Detection/validation/validation/NORMAL/NORMAL-12494-33.jpeg)

