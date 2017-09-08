
# GOOWE
**August 2017: Our paper is accepted for publication in ACM Transactions on Knowledge Discovery from Data (TKDD)

GOOWE: Geometrically Optimum and Online-Weighted Ensemble Classifier for Evolving Data Streams

This project is for sharing our implementation of GOOWE ensemble classifiers for evolving data streams. 

Abstract: 
Designing adaptive classifiers for an evolving data stream is a challenging task due to its size and dynamically changing nature. Combining individual classifiers in an online setting, the ensemble approach, is one of the well-known solutions. It is possible that a subset of classifiers in the ensemble outperforms others in a time-varying fashion. However, optimum weight assignment for component classifiers is a problem which is not yet fully addressed in online evolving environments. We propose a novel data stream ensemble classifier, called Geometrically Optimum and Online-Weighted Ensemble (GOOWE), which assigns optimum weights to the component classifiers using a sliding window containing the most recent data instances. We map vote scores of individual classifiers and true class labels into a spatial environment. Based on the Euclidean distance between vote scores and ideal-points, and using the linear least squares (LSQ) solution, we present a novel dynamic and online weighting approach. While LSQ is used for batch mode ensemble classifiers, it is the first time that we adapt and use it for online environments by providing a spatial modeling of online ensembles. In order to show the robustness of the proposed algorithm, we use real-world datasets and synthetic data generators using the MOA libraries. First, we analyze the impact of our weighting system on prediction accuracy through two scenarios. Second, we compare GOOWE with 8 state-of-the-art ensemble classifiers in a comprehensive experimental environment. Our experiments show that GOOWE provides improved reactions to different types of concept drift compared to our baselines. The statistical tests indicate a significant improvement in accuracy, with conservative time and memory requirements. 


Webpage: https://hamedrab.github.io/GOOWE/