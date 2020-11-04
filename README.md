# MiNgMatchSegmenter

MiNgMatch Segmenter is a data-driven word segmentation algorithm searching for the shortest sequence of n-grams needed to match the input string.
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
