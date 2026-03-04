rm -rf build
rm -rf mlp.engine
mkdir build
cd build
cmake .. && make
./mlp -s
./mlp -d
cd ..