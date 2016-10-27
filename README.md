# RBM
Restricted Boltsmann Machine
## default format
python rbm.py digitstrain.txt digitsvalid.txt digitstest.txt 
## specify parameter
please run python rbm.py --help for details
## pretraining
run rbm.py or autoencoder.py, it will generate a p.pickle file
Then run BasicNN.py, it will take p.pickle 
You can also use ae.pickle or rbm.pickle for precomputed parameters (need to modify BasicNN.py)
## denoising AE
specify "-d 0.5" when running autoencoder.py
e.g python autoencoder.py digitstrain.txt digitsvalid.txt digitstest.txt  -d 0.5

Thanks for reading and please contact shiyud@andrew.cmu.edu for any issue.


