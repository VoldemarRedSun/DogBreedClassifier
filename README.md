В данной работе построены модели нейронных сетей  для многоклассовой классификации.
В качестве обущающих данных используется Stanford Dogs Dataset, содержащий 120 пород собак и примерно по 100 фотографий для каждой породы.
Исходный источник данных находится по адресу http://vision.stanford.edu/aditya86/ImageNetDogs/ 
и содержит дополнительную информацию о разделении тренировок/тестов и исходных результатах.
Данный датасет связан со следующими статьями:
1) Aditya Khosla, Nityananda Jayadevaprakash, Bangpeng Yao and Li Fei-Fei. Novel dataset for Fine-Grained Image Categorization. First Workshop on Fine-Grained Visual Categorization (FGVC), IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2011. [pdf] [poster] [BibTex]
2) J. Deng, W. Dong, R. Socher, L.-J. Li, K. Li and L. Fei-Fei, ImageNet: A Large-Scale Hierarchical Image Database. IEEE Computer Vision and Pattern Recognition (CVPR), 2009. [pdf] [BibTex]

Так же данный датасет можно скачать через Kaggle https://www.kaggle.com/datasets/jessicali9530/stanford-dogs-dataset.

В данной работе в качестве модели нейронной сети используется ансамбли из трех дообученных нейронных сетей Resnet50. Лучше всего проявил себя ансамбль, построенный с помощью усреднения предсказаний трех 
 моделей Resnet50. Он дает accuracy 83 процента, при этом каждый resnet50 обучался 3 эпохи, обучающие выборки были построены с помощью бутстрэпа. Ансамбль, который учитывает предсказания resnet c помощью полносвязного слоя обучается нейстойчиво и долго, в конце результат получается хуже чем у ансамбля с усреднением. Так же была написана своя архитектура сверточной нейронной сети.
Формат обучающих данных - директория с папками, в каждой папке лежат фотографии с одной породой собак.  
