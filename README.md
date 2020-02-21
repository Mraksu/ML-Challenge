
# Machine Learning Challenge


This is the code used for Machine Learning Challenge (Speech Recognition) at MSc Data Science and Society Programme at Tilburg University.

We have finished 3rd place.

**Introduction**

In this challenge, the goal is to learn to recognise which English word was spoken in an audio recording. Recognising phonemes to be able to recognise an arbitrary word is not the task at hand. Rather, it is limited to recognising a set number of words. This is a multinomial classification problem.

Preprocessing and extracting features
Although various feature extraction methods exist, mel-frequency cepstral coefficients (MFCCs) seem to be the standard for speech recognition tasks, so we used the given features (Alim & Rashig, 2018).
	The features of each audio file exists as a matrix representing 13 coefficients over a number of frames. In order to deal with the different number of frames, the shorter features were padded with zeros at the end to match the size of the largest features in the dataset. Additionally, a total of 6 incorrect features were discarded.
	We have standardised the data before usage, as some learning algorithms are sensitive to the scale of the data. Multiple feature transformation methods were attempted, such as minima, maxima and means. However, these produced overall poor results and this prompted us to simply flatten the features, except for the RNNs, to be able to feed them to the models.
	The given classes were encoded and subsequently one hot encoded, in order to ensure that a higher categorical value was not perceived as better by the model.
	The features were split into training and testing sets. Of the training data, 20% was reserved for validation. For this, labels were stratified, in order not to lose the balance in the dataset. Only after the desired model was found based on validation, the model in question was trained with all of the training data.

**Learning algorithms**

We experimented with a number of learning algorithms: Recurrent Neural Networks (RNNs), Multilayer Perceptrons (MLPs), Support Vector Machines (SVMs), Decision Trees (DTs) and Logistic Regressions (LRs). We were expecting the NNs and SVMs to perform best due to their persistent use in the scientific community, but we nevertheless decided to compare their performance against the more traditional algorithms. After testing multiple algorithms, we decided to work with an RNN, more specifically Long Short-Term Memory (LSTM), due to its ability to deal with sequences and their quick convergence (Sak et al., 2014).

**Hyperparameter tuning**

For LR, SVMs and DTs, we used the default Scikit Learn parameters. Most of our experimentation was done with NN, with a focus on  LSTMs.  We have tried various number of hyperparameters such as the number of hidden layers and neurons, batch sizes, optimizers, activation functions and epochs. We finally went with an LSTM model, using a (recurrent) dropout of 0.2 to prevent overfitting. We used the adam optimizer, used a batch size of 300 and trained for 40 epochs. See Table 1 for the final model architecture.

**Results**

The traditional methods did not perform well. MLPs started to offer more desired accuracies, but did not get higher with more parameter tuning. In the end, the LSTM model performed best, giving us a 95.14% accuracy score on CodaLab.

 
**References**

Alim, S. A., Rashid, N. K. A. (2018). Some commonly used speech feature extraction algorithms. From Natural to Artificial Intelligence-Algorithms and Applications.

Sak, H., Senior, A., & Beaufays, F. (2014). Long short-term memory based recurrent neural network architectures for large vocabulary speech recognition. arXiv preprint arXiv:1402.1128.




