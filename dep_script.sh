#!/bin/bash
# Assume conda installation

echo "==>dependencies setup for deep_q_rl"

echo "==>updating current package..."
sudo apt-get update

echo "==>Installing bunch of dependencies through conda"
conda install opencv matplotlib tk numpy ipython nose scipy six

echo "==>installing Theano ..."
# some dependencies ...
sudo apt-get install python-dev python-pip g++ libopenblas-dev git
# pip install --user --upgrade --no-deps git+git://github.com/Theano/Theano.git
# Should get Theano straight from source
pip install git+https://github.com/Theano/Theano.git
pip install git+https://github.com/Lasagne/Lasagne.git

echo "==>installing Lasagne ..."
pip install --user --upgrade https://github.com/Lasagne/Lasagne/archive/master.zip

# Packages below this point require downloads. 
mkdir build
cd build

if [ ! -d "./pylearn2" ]
then
echo "==>installing Pylearn2 ..."
# dependencies...
sudo apt-get install libyaml-0-2
git clone git://github.com/lisa-lab/pylearn2
fi
cd ./pylearn2
python setup.py develop --user
cd ..

if [ ! -d "./ALE" ]
then
echo "==>installing ALE ..."

# dependencies ...
sudo apt-get install libsdl1.2-dev libsdl-gfx1.2-dev libsdl-image1.2-dev cmake

git clone https://github.com/mgbellemare/Arcade-Learning-Environment ALE
cd ./ALE
cmake -DUSE_SDL=ON -DUSE_RLGLUE=OFF .
make -j2
pip install --user .
cd ..
fi

echo "==>All done!"
