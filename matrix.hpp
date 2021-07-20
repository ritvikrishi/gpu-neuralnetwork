#include<bits/stdc++.h>
#include <random>
#include<chrono>
#define pb push_back
using namespace std;

class Matrix{
public:
    int rows, cols;
    vector<vector<double>>vec;
    Matrix(int rows, int cols, string s, int sd);
    Matrix* add(Matrix* m2, bool usegpu);
    Matrix* multiply(Matrix* m2, bool usegpu);
    Matrix* subtract(Matrix* m2, bool usegpu);
    double vecdot(Matrix* m1, Matrix* m2, bool usegpu);
    Matrix* matmul(Matrix* m2, bool usegpu);
    Matrix* T(bool usegpu);
    Matrix* square(bool usegpu);
    Matrix* msqrt(bool usegpu);
    Matrix* elinv(bool usegpu);
    Matrix* cmult(double val, bool usegpu);
    Matrix* cadd(double val, bool usegpu);
    void print();
    void shape();
    bool compare(Matrix *m2);
    Matrix* reshape(int rows, int cols);
};
Matrix::Matrix(int rows, int cols, string s, int val){
    this->rows = rows;
    this->cols = cols;
    if(s == "rand"){
        unsigned long long int now = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
        default_random_engine generator(now%10000);
        normal_distribution<double> distribution(0, sqrt(1.0/val));
        for(int i = 0; i < rows; i++){
            this->vec.pb({});
            for(int j = 0; j < cols; j++){
                this->vec[i].pb(distribution(generator)); // what scale
            }
        }
    }
    else if(s == "fixed"){
        for(int i = 0; i < rows; i++){
            this->vec.pb({});
            for(int j = 0; j < cols; j++){
                this->vec[i].pb(val);
            }
        }
    }
    
}

