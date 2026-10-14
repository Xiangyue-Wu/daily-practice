# Daily Practice

A collection of AI and algorithm practice projects, organized by technical domain. This monorepo covers data processing, classical machine learning, deep learning, computer vision, natural language processing, reinforcement learning, computational intelligence, robotics, and small AI systems.

## Project Categories

1. [Data Processing & Mining](01-data-processing-and-mining/)
2. [Classical Machine Learning](02-classical-machine-learning/)
3. [Deep Learning](03-deep-learning/)
4. [Computer Vision & Multimodal Perception](04-computer-vision-and-multimodal/)
5. [Natural Language Processing](05-natural-language-processing/)
6. [Reinforcement Learning & Decision Making](06-reinforcement-learning/)
7. [Computational Intelligence & Optimization](07-computational-intelligence/)
8. [Robotics & Spatial Perception](08-robotics-and-spatial-perception/)
9. [AI Systems & Deployment](09-ai-systems-and-deployment/)

## Project Index

### 1. Data Processing & Mining

- [Advanced Data Mining](01-data-processing-and-mining/advanced-data-mining/) — Time-series analysis, anomaly detection, and MLP forecasting for household power data.
- [Movie Data Cleaning and Visualization](01-data-processing-and-mining/data-cleaning-and-visualization/) — Tabular cleanup, feature preparation, exploratory plots, and rating-model experiments.
- [Retail Data Exploration and Preprocessing](01-data-processing-and-mining/data-exploration-and-preprocessing/) — Retail-data cleaning, customer similarity analysis, scaling, encoding, and export.
- [Retail Association Mining and Customer Clustering](01-data-processing-and-mining/unsupervised-learning-clustering/) — Apriori rules, RFM features, K-means, PCA, and hierarchical clustering.

### 2. Classical Machine Learning

- [Decision Tree Implementations](02-classical-machine-learning/decision-trees/) — Manual and library-based decision-tree workflows for categorical classification.
- [Bagging and Boosting with Decision Stumps](02-classical-machine-learning/ensemble-learning/) — Baseline, bagging, and AdaBoost-style classifiers for tabular data.
- [Generative Classifiers on Mixed-Type Data](02-classical-machine-learning/generative-classifiers/) — Naive Bayes, Gaussian discriminant analysis, and EM-based mixture classification.
- [Wine Quality Regression and Classification](02-classical-machine-learning/supervised-learning-regression/) — Linear regression and classical classifiers with evaluation and parameter search.

### 3. Deep Learning

- [Battery State-of-Health Prediction](03-deep-learning/battery-soh-prediction/) — Feature preparation, LSTM baselines, and CNN–BiLSTM–attention transfer workflows.
- [CIFAR-10 Image Classification](03-deep-learning/cifar10-image-classification/) — Compact CNN training, checkpoint evaluation, and per-class visualization.
- [Semi-Supervised Radar Image Classification](03-deep-learning/deep-semi-supervised-learning/) — ResNet-18 baselines and a FixMatch-style limited-label workflow for MSTAR imagery.
- [Handwritten Digit Recognition](03-deep-learning/handwritten-digit-recognition/) — Distance, BP network, and CNN approaches retained as separate implementations.
- [Neural Networks for MNIST and CIFAR-10](03-deep-learning/neural-networks-and-deep-learning/) — CNN, AlexNet, checkpoint fine-tuning, and cross-dataset adaptation experiments.
- [Skeleton-Based Hand Action Recognition](03-deep-learning/skeleton-based-action-recognition/) — LSTM, residual temporal convolution, and Transformer models for skeleton sequences.

### 4. Computer Vision & Multimodal Perception

- [Face Recognition and Reconstruction with PCA](04-computer-vision-and-multimodal/face-recognition-and-reconstruction/) — Eigenface-style projection, nearest-match recognition, and face reconstruction.
- [License Plate Digit Recognition](04-computer-vision-and-multimodal/license-plate-digit-recognition/) — Character segmentation, handcrafted features, and classifier-based digit recognition.
- [Temperature and Image Fusion for Fire Detection](04-computer-vision-and-multimodal/multimodal-temperature-image-fusion/) — CLIP features and FiLM-conditioned visual models for paired sensor and image data.
- [Image Contrast Enhancement and Dehazing](04-computer-vision-and-multimodal/visual-image-enhancement/) — CLAHE contrast enhancement and dark-channel-prior haze removal.

### 5. Natural Language Processing

- [Advanced Text Processing Utilities](05-natural-language-processing/advanced-text-processing/) — TF-IDF similarity, TF-IDF/TextRank keywords, and Levenshtein correction.
- [Chinese Text Collection and Basic Processing](05-natural-language-processing/basic-text-processing/) — Text extraction, cleanup, jieba segmentation, frequency analysis, and word clouds.
- [Chinese Sentiment Classification](05-natural-language-processing/machine-learning-for-nlp/) — Logistic regression, TextCNN, Bi-LSTM, and repeated-seed stability analysis.
- [Chinese NLP Case Studies](05-natural-language-processing/nlp-case-studies/) — Cleaning, visualization, classical sentiment analysis, SnowNLP, and ALBERT workflows.

### 6. Reinforcement Learning & Decision Making

- [Cliff Walking](06-reinforcement-learning/cliff-walking/) — Dynamic programming, SARSA, and Q-learning on a tabular gridworld.
- [Jack's Car Rental](06-reinforcement-learning/jacks-car-rental/) — Policy and value iteration for a stochastic vehicle-transfer problem.

### 7. Computational Intelligence & Optimization

- [Ant Colony and Artificial Immune Algorithms](07-computational-intelligence/ant-colony-and-immune-algorithms/) — Ant-colony variants and immune-inspired optimization and detection methods.
- [Evolutionary Channel Pruning for ResNet-32](07-computational-intelligence/computational-intelligence-model-compression/) — Genetic and particle-swarm searches for structured network compression.
- [Genetic Algorithms for Optimization](07-computational-intelligence/optimization-and-genetic-algorithms/) — Continuous, traveling-salesperson, and knapsack optimization examples.
- [Particle Swarm and Artificial Bee Colony Optimization](07-computational-intelligence/particle-swarm-and-bee-colony/) — Continuous, binary, and permutation-based swarm optimization workflows.

### 8. Robotics & Spatial Perception

- [Stereo Disparity Estimation](08-robotics-and-spatial-perception/depth-estimation-and-slam/) — OpenCV stereo block matching for disparity-map generation.
- [Robot Workspace Safety Check](08-robotics-and-spatial-perception/robot-kinematics-simulation/) — A focused MATLAB utility for validating positions against workspace limits.

### 9. AI Systems & Deployment

- [Intelligent Sentiment Analysis Web Application](09-ai-systems-and-deployment/intelligent-sentiment-analysis/) — A Flask interface combining Transformers, SnowNLP, uploads, and batch summaries.

## Reproduction

Each project README includes a project-specific prompt for reproducing the available workflow with an AI coding assistant. Some projects require separately obtained datasets, model weights, pretrained resources, or compatible environments. Inclusion in this repository means the source and documentation passed static and public-safety checks; it does not mean that every project has been fully executed in the published form.
