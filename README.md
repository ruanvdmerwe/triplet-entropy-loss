# Triplet Entropy Loss

## Abstract
Spoken language identification systems form an integral part in many speech recognition tools today. Over the years many techniques have been used to identify the language spoken, given just the audio input, but in recent years the trend has been to use end to end deep learning systems. Most of these techniques involve converting the audio signal into a spectrogram which can be fed into a Convolutional Neural Network which can then predict the spoken language. This technique performs very well when the data being fed to model originates from the same domain as the training examples, but as soon as the input comes from a different domain these systems tend to perform poorly. Examples could be when these systems were trained on WhatsApp recordings but are put into production in an environment where the system receives recordings from a phone line. 

The research presented investigates several methods to improve the generalisation of language identification systems to new speakers and to new domains. These methods involve Spectral augmentation, where spectrograms are masked in the frequency or time bands during training and by using CNN architectures that are pre-trained on the Imagenet dataset. The research also introduces the new Triplet Entropy Loss training method. This training method involves training a network simultaneously using Cross Entropy and Triplet loss. Several tests were run with three different CNN architectures to investigate what the effect all three of these methods have on the generalisation of a LID system.

The tests were done in a South African context on six languages, namely Afrikaans, English, Sepedi, Setswanna, Xhosa and Zulu. The two domains tested were data from the NCHLT speech corpus, used as the training domain, with the Lwazi speech corpus being the unseen domain. 

It was found that all three methods improved the generalisation of the models, though not significantly. Even though the models trained using Triplet Entropy Loss showed a better understanding of the languages and higher accuracies, it appears as though the models still memorise word patterns present in the spectrograms rather than learning the finer nuances of a language. The research shows that Triplet Entropy Loss has great potential and should be investigated further, but not only in language identification tasks but any classification task.

The full paper can be found [here](https://arxiv.org/abs/2012.03775)

## TEL overview

For tasks such as language identification where the input data can contain data which is present in many other classes as well, such as someone speaking a mix of words from different languages, it will be more optimal to have a loss function that interprets interactions between classes at the output. The loss must optimize the network by learning these interactions between classes to generalize better to the instances where there is a tiny threshold between the classes. A loss that loosely fits this description is the Triplet loss function. By using the Triplet loss function, the weights of a network are being optimized by comparing different classe embeddings with one another and optimizing the distance between the embeddings such that different classes are far from one another. The model can then learn special characteristics of all the classes and in doing so could be able to better learn the fine connections between languages such as English and Zulu for instance. But Triplet loss does not optimize for prediction capabilities directly.

The TEL training method aims to leverage both the strengths of Cross Entropy Loss (CEL) and Triplet loss during the training process, assuming that it would lead to better generalization for language identification tasks. The TEL method though does not contain a pre-training step, but trains simultaneously with both CEL and Triplet losses, as shown in the figure below. As seen, the final embedding layer feeds into two separate layers where each of these output layers are connected to two different losses. TEL can be represented by the equation.

![network_image](https://github.com/ruanvdmerwe/triplet-entropy-loss/blob/main/images/tel_model.jpg)

![formula](https://github.com/ruanvdmerwe/triplet-entropy-loss/blob/main/images/tel_eq.PNG)

Below is the embeddings generated form a Densenet121 model trained with TEL, CEL and Triplet loss.

![formula](https://github.com/ruanvdmerwe/triplet-entropy-loss/blob/main/images/densenet_embeddings.png)

## Using the repo

### Getting the data

In order to dowload the South African speech language used in the project, please visit [Sadilar](https://repo.sadilar.org/handle/20.500.12185/7) where the NCHLT, NCHLT Auxliary and Lwazi data can be downloaded. The data can then be extracted to your working directory.

### Code

All of the code used for this project can be found in the src folder. 

The **data_handling.py** file contains all of the code required to manipulate the extracted data for all of the speech data. This involves cleaning as well as converting the speech to the correct spectrogram format.

The **train_models.py** files contain code to train a custom model, Inception-V3 model, Resnet50 model as well as a Densenet121 model. If you want to use pre-trained weights ensure to change the weights to be loaded to 'imagenet' in all of the return_{architecture}_model functions. The code will run three training rounds for each model and save that specific model's weights as well as the training history.

To generate embeddings from from the models created, you can use the **create_embeddings.py** file.

The folder also contains a file, **spectobot.py** which can be used to launch a telegram application that can be used to further gather data about model performances in the wild as well as help in gathering data to further train models. Specifically on utterances where users speak normally instead of reading of a prompt.


## References and Notes

If you use TEL or this code, please reference:

@article{van2020triplet,
  title={Triplet Entropy Loss: Improving The Generalisation of Short Speech Language Identification Systems},
  author={van der Merwe, Ruan},
  journal={arXiv preprint arXiv:2012.03775},
  year={2020}
}
