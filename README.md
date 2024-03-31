# promoter-sequence-transcription-rates

All models trained on the data from ["Automated model-predictive design of synthetic promoters to control transcriptional profiles in bacteria"](https://www.nature.com/articles/s41467-022-32829-5) (LaFleur et al.).

This model decreased mean squared error from 0.251 to 0.143 (by 43.0%) using a Long Short-Term Memory model trained on:
* The upstream 6-nucleotide site called the −35 motif
* The downstream 6-nucleotide site called the −10 motif
* The nucleotide region that appears upstream of the −35 motif, called the UP element
* The spacer region that separates the −10 and −35 motifs
* A typically, 6-nucleotide region in between the −10 motif and TSS, called the discriminator (Disc)
* The first 20 transcribed nucleotides, called the initial transcribed region (ITR).

While the first two features have fixed lengths (both six base pairs), the last four features vary in length. Adding padding to the data to include all sequences decreased the MSE by 14.0%. Padding makes all the inputs equal in length by adding layers of zeros or other "filler" data outside the actual data in an input matrix. The primary purpose of padding is to preserve the spatial size of the input so that the output, after applying filters (kernels), remains the same size or adjusts according to the desired output dimensions (deepai.org, n.d.). 

According to Reddy and Reddy (2019), there is little difference in performance between pre- and post-padding in LSTMs, unlike with CNNs. However, they did find that LSTM pre-padding was marginally more accurate (5), so the padding will go upstream (before) the data. For a more comprehensive comparison, see "padding_comparison.ipynb". Additionally, I excluded the spacer sequence ('spacs' column) with lengths other than 16, 17, or 18. The other sequences have been synthetically developed and vary in size from 0 to 31. This large standard range does not help produce more accurate results and only increases runtime. The figure below shows the distribution of lengths for the spacer sequence. Excluding the synthetic spacer sequence decreased the sample size by 508, from 13,842 to 13,334.

![distribution of spacer sequence lengths](supporting_documents/image.png)

A CNN would be less well-suited for this dataset compared to an LSTM because CNNs are primarily used for grid-structured data like images. Although they can also be applied to sequential data by treating the data as a one-dimensional grid (e.g., for text classification tasks), CNNs do not inherently capture sequential dependencies as effectively as LSTMs (O'Shea and Nash, 2015). This aligned with my findings, which showed that a CNN trained on the −35 motif and −10 motif (fixed length inputs) had a 12% higher MSE compared to a LSTM trained on the same data. Lastly, I will also compare the models to LaFleur et al.'s MLR, which encodes the presence of each possible three-nucleotide sequence within each hexamer (12).

I was initially going to compare my LSTM to an MLR that one-hot encodes each of the six inputs as a different classification. However, this approach treats every unique sequence in each column as a separate categorical input. With 5 rows and 11942 columns, fitting an MLR model to this data would take an unrealistic amount of time and be very inaccurate. For this code, see "MLR.ipynb".

## Prediction Tool

"pred_tool.ipynb" has forward- and backward-compatible prediction tools. The forward-compatible prediction tool predicts the transcription rate of a novel sequence. Given the UP, h35, spacs, h10, disc, and ITR promoter sequences (str[]) and the model path (str), the tool returns the predicted transcription rate (float)

Meanwhile, the backward-compatible prediction tool predicts the promoter sequences closest to a given target. This applies a combinatorial approach. It simulates all possible combinations of promoter sequences (not specified), encodes them, and predicts the transcription rate. The closest predictions to the target are returned. Alternative approaches include an inverse transformation, specialized model, and database. These are more efficient. However, they are not exhaustive models and will exclude sequences.

The model takes in the required parameters:
* The path to the model (str)
* Target value to predict (float)

It also takes the following optional parameters:
* The maximum difference between the predicted value and the target (float)
* Number of results to return (int)
* Maximum number of iterations (int)
* The UP, h35, spacs, h10, disc, and ITR (str) promoter sequences required in the output

It returns a list of dictionaries with the results of each prediction. Each dictionary contains:
* The predicted value (float)
* The difference between the predicted and the target (float)
* The promoter sequences for UP, h35, spacs, h10, disc, ITR (str[])

All internals are delegated to "pred_tool_calc.py".

## Work Cited

Brownlee, Jason. “Hyperparameter Optimization with Random Search and Grid Search.” MachineLearningMastery.Com, 18 Sept. 2020, machinelearningmastery.com/hyperparameter-optimization-with-random-search-and-grid-search/.

LaFleur, Travis L., et al. “Automated Model-Predictive Design of Synthetic Promoters to Control Transcriptional Profiles in Bacteria.” Nature News, Nature Publishing Group, 2 Sept. 2022, www.nature.com/articles/s41467-022-32829-5.

“Padding (Machine Learning).” DeepAI, DeepAI, 17 May 2019, deepai.org/machine-learning-glossary-and-terms/padding.

Reddy, Mahidhar Dwarampudi, and Subba Reddy. “Effects of Padding on Lstms and CNNS.” arXiv.Org, 18 Mar. 2019, arxiv.org/abs/1903.07288.

Russell, Peter J. IGenetics. Benjamin Cummings, 2006.
