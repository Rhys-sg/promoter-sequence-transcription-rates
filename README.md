# promoter-sequence-transcription-rates

Using the data from "Automated model-predictive design of synthetic promoters to control transcriptional profiles in bacteria" (LaFleur et al.).

https://www.nature.com/articles/s41467-022-32829-5

Decreased MSE from 0.250 to 0.177 (29.2%) using a convolutional neural network based on the:
•	upstream 6-nucleotide site called the −35 motif
•	downstream 6-nucleotide site called the −10 motif
This does not include sequences that vary in length:
•	a 20-nucleotide region that appears upstream of the −35 motif, called the UP element
•	a spacer region that separates the −10 and −35 motifs
•	a typically 6-nucleotide region in between the −10 motif and TSS, called the discriminator (Disc)
•	or the first 20 transcribed nucleotides, called the initial transcribed region (ITR).
To include the sequences that vary in length, I will need to change the model from a CNN to a many-to-one RNN.

