Long Short Term Memory Units (original README)
============================
This is self-contained package to train a language model on word level Penn Tree Bank dataset. 
It achieves 115 perplexity for a small model in 1h, and 81 perplexity for a big model in 
a day. Model ensemble of 38 big models gives 69 perplexity.
This code is derived from https://github.com/wojciechz/learning_to_execute (the same author, but 
a different company).


More information: http://arxiv.org/pdf/1409.2329v4.pdf

For the Deep Learning NYU spring 2015 course
==========================
Modifications to the original code:

+ Made functions global and put the main part outside of a function, for easier interactive sessions.
+ Added a4\_commununication\_loop.lua for an example of stdin/stdout communication.
+ Added character-preprocessed train and validation ptb set in data/.
+ Modified data.lua so we can all easily load the data in the same way and agree on the dictionary. 
+ Added a simple script a4\_vocab.lua that loads the data and prints the character-level vocabulary (which is the vocabulary that will also be used in grading).
+ Added a4\_grading.py so you can test how your program performance will be automatically evaluated.

For more information, see the assignment instructions pdf.
