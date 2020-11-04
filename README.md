# MiNgMatchSegmenter

MiNgMatch Segmenter is a fast word segmentation algorithm searching for the shortest sequence of n-grams needed to match the input string.
It has been designed for and tested on the Ainu language (a critically endangered language native to the northern parts of the Japanese archipelago) but can be applied to any other language, as well.

## Usage

Assuming that your n-gram data is stored in `ngrams.txt` and `input.txt` is the text you want to process, you can use the segmenter as follows:

```
python MiNgMatchSegmenter.py -t ngrams.txt -i input.txt -o output.txt
```
You can specify the maximum number of n-grams allowed per input segment, like this:
```
python MiNgMatchSegmenter.py -t ngrams.txt -i input.txt -o output.txt -l 2
```
The n-gram data should be provided as a tab-delimited file of the following format:
```
koranan   5,3 -3.34220648264453
penekusu  4,2 -3.3427874786936
korwa     3   -3.3439518077607
patek         -3.34628987252967
```
The first column stores unsegmented strings used by the matching algorithm, each of them corresponding to a lexical n-gram.
The second column are the indices of word boundaries for each n-gram (sorted in descending order).
The rightmost column contains the score for each n-gram (proportional to its frequency), represented as logarithm.


## Citation
When using the code, please cite the following paper:

```
@article{Nowakowski_2019,
title={MiNgMatch—A Fast N-gram Model for Word Segmentation of the Ainu Language},
author={Nowakowski, Karol and Ptaszynski, Michal and Masui, Fumito},
journal={Information},
volume={10},
url={http://dx.doi.org/10.3390/info10100317},
DOI={10.3390/info10100317},
number={10},
year={2019},
month={Oct},
pages={317}
}
```
