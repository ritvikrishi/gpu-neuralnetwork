#include <bits/stdc++.h>
#include "matrix.hpp"
class Layers{
public: 
    string layer_type;
};

class ffnn : public Layers{
public:
    vector<Matrix*>W;
    vector<Matrix*>B;
    vector<Matrix*>A;
    vector<Matrix*>H;
    vector<Matrix*>dw;
    vector<Matrix*>db;
    vector<Matrix*>da;
    vector<Matrix*>dh;
    int num_input;
    int num_output;
    int num_layers;
    int num_neurons; 
    string activation_func;
    string loss_func;
    bool usegpu;
    Matrix* g(Matrix* Ai);
    Matrix* o(Matrix* Al);
    Matrix* e(int index);
    Matrix* g1(Matrix* Ai);
    ffnn(int num_input, int num_output, int num_layers, int num_neurons);
    void backprop(Matrix* output, int index);
    Matrix* forwardprop(int index);
    double loss(Matrix* output, int index);
    void accuracy(string s);
    void train(string optimizer, int num_training, int num_epochs, int batch_size, double learning_rate);
};
