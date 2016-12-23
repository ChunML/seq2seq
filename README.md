A simple project to create a language translation model using Sequence To Sequence Learning Approach.

Details can be found at my blog post here:

* [Creating A Language Translation Model Using Sequence To Sequence Learning Approach](https://chunml.github.io/ChunML.github.io/project/Sequence-To-Sequence/)

### Dataset

I used the Europarl's Parallel Corpus for training. To get the source code to work immediately, you have to use the newest version (release v8 at the time of writing) at the link below (following the link will start a 180MB download):

* [Europarl v8](http://www.statmt.org/wmt15/europarl-v8.fi-en.tgz)

Feel free to change the default dataset to anyone of your own. Just don't forget to modify the code!

### List of arguments:

* max_len: specify the maximum length of sentence to extract from text.  
Default: 200
* vocab_size: specify the number of the most frequent words to put in vocabulary set.  
Default: 20000
* batch_size: specify the batch size.  
Default: 1000
* layer_num: specify the number of recurrent layers in Encoder network.  
Default: 3
* hidden_dim: specify the dimension of hidden state.  
Default: 1000
* np_epoch: specify the number of training epochs.  
Default: 20
* mode: specify whether to train or test the model.  
Default: train

### Train the model:

* With default settings:

```python
python seq2seq.py
```

* With user-defined settings:

```python
# Max length:= 300, number of recurrent layers:= 2, dimension of hidden state:= 500
python seq2seq.py -max_len 300 -layer_num 2 -hidden_dim 500
```

### Test the model:

The network must be trained at least once (trained weights must exist!).

* If the network was trained with default settings:

```python
python seq2seq.py -mode test
```

* If the network was trained with user-defined settings:

```python
# Max length:= 300, number of recurrent layers:= 2, dimension of hidden state:= 500
python seq2seq.py -mode test -max_len 300 -layer_num 2 -hidden_dim 500
```
