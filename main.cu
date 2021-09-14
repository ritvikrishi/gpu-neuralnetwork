//////////////////////////////////////////////////////////
//                                                      //
//          GPU COURSE PROJECT - 2021                   //
//                                                      //
//   AIM :- Implementing a neural network in GPU and    // 
//          comparing the times with that of CPU.       //
//                                                      //
//   COLLABORATORS :-                                   //
//              SUDHEENDRA - CS18B006                   //
//              RITVIK RISHI - CS18B057                 //
//                                                      //
//   DATE :- 09-05-2021                                 //
//                                                      //
//////////////////////////////////////////////////////////

#include <bits/stdc++.h>
#include "layers.hpp"
#include <cmath>
#include <cuda.h>
#include <sys/time.h> 
using namespace std;

// Error check in GPU
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true){
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

// To store the time values at various instances.
struct timeval t1, t2, t3, t4;

// Total cpu function execution time.
double cpu_function_time = 0;

// Total GPU function execution time.
double gpu_function_time = 0;

// Total GPU kernel execution time.
double gpu_kernel_time = 0;

// USING MNIST DATASET.
// Training Image data.
FILE* training_img = fopen("data/train_image.txt", "r");

// Training Label data.
FILE* training_label = fopen("data/train_label.txt", "r");

// Validation Image data.
FILE* validation_img = fopen("data/validation_image.txt", "r");

// Validation Label data.
FILE* validation_label = fopen("data/validation_label.txt", "r");

// Test Image data.
FILE* test_img = fopen("data/test_image.txt", "r");

// Test Label data
FILE* test_label = fopen("data/test_label.txt", "r");

// Storing the file data in variables
double* x_train = new double[784 * 50000];
int* y_train = new int[50000];
double* x_test = new double[784 * 10000];
int* y_test = new int[50000];
double* x_val = new double[784 * 10000];
int* y_val = new int[50000];

// Kernel to add two matrices, number of threads = number of elements in the matrix.
__global__ void gpu_add(double* arr1, double* arr2, double* ans, int len){
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if(index < len){
        ans[index] = arr1[index] + arr2[index];
    }
}

// Kernel to subtract two matrices, number of threads = number of elements in the matrix.
__global__ void gpu_subtract(double* arr1, double* arr2, double* ans, int len){
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if(index < len){
        ans[index] = arr1[index] - arr2[index];
    }
}

// Kernel to compute hadamard of two matrices, number of threads = number of elements in the matrix.
__global__ void gpu_mult(double* arr1, double* arr2, double* ans, int len){
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if(index < len){
        ans[index] = arr1[index] * arr2[index];
    }
}

// Kernel to perform matrix multiplication, no of threads = number of elements in result.
__global__ void gpu_matmul(double* a, double* b, double* c, int m, int n, int k){
	int row = blockIdx.y * blockDim.y + threadIdx.y; 
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    double sum = 0;
    if(col*row < m*k && col<k && row<m){
        for(int i = 0; i<n; i++){
            sum += a[row * n + i] * b[i * k + col];
        }
        c[row * k + col] = sum;
    }
}

// Kernel to compute scaler inner product of two vectors, DEPRECATED.
__global__ void gpu_vecdot(double* arr1, double* arr2, double* ans, int len){
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if(index < len){
		ans[0] += arr1[index]*arr2[index];
	}
}

// Kernel to compute transpose of a matrix, no of threads = no of elements in matrix.
__global__ void gpu_T(double* arr1, double* ans, int rows, int cols){
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if(index < rows*cols){
		int i = index / cols;
		int j = index % cols;
		ans[j*rows + i] = arr1[index];
	}
}

// kernel to compute square of a matrix, no of threads = no of elements in matrix.
__global__ void gpu_square(double* arr1, double* ans, int len){
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if(index < len){
        ans[index] = arr1[index]*arr1[index];
    }
}

// Kernel to compute sqrt of a matrix, no of threads = no of elements in matrix.
__global__ void gpu_sqrt(double* arr1, double* ans, int len){
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if(index < len){
        ans[index] = sqrt(arr1[index]);
    }
}

// Kernel to invert every element in a matrix, no of threads = no of elements in matrix.
__global__ void gpu_elinv(double* arr1, double* ans, int len){
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if(index < len){
        ans[index] = 1.0/(arr1[index]);
    }
}

// Kernel to add a constant to every element of matrix, no of threads = no of elements of matrix.
__global__ void gpu_cadd(double* arr1, double* ans, double val, int len){
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if(index < len){
        ans[index] = arr1[index]+val;
    }
}

// Kernel to multiply a constant to every element of matrix, no of threads = no of elements of matrix.
__global__ void gpu_cmult(double* arr1, double* ans, double val, int len){
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if(index < len){
        ans[index] = arr1[index] * val;
    }
}

// Get gpu pointer corresponding to the cpu pointer.
template <typename T>
T* get_gpu(T* temp, int len){
    T* gpu_temp;
    cudaMalloc(&gpu_temp, len * sizeof(T));
    cudaMemcpy(gpu_temp, temp, len * sizeof(T), cudaMemcpyHostToDevice);
    return gpu_temp;
}

// get cpu pointer corresponding to the gpu pointer
template <typename T>
T* get_cpu(T* temp, int len){
    T* cpu_temp = new T[len];
    cudaMemcpy(cpu_temp, temp, len * sizeof(T), cudaMemcpyDeviceToHost);
    return cpu_temp;
}

// copies all the matrices in the vector<Matrix*> and returns the copy
vector<Matrix*> deepcopy(vector<Matrix*> &temp){
	vector<Matrix*> ans;
    ans.pb({});
	for(int i=1; i < temp.size(); i++){ 
		Matrix* t1 = temp[i]->cadd(0.0, false);
		ans.pb(t1);
	}
	return ans;
}

// get a double* corresponding to the vec from the Matrix object
double* vec_to_arr(Matrix m){
    int rows = m.rows;
    int cols = m.cols;
    double* ans = new double[rows * cols];
    for(int i = 0; i < rows; i++){
        for(int j = 0; j < cols; j++){
            ans[i * cols + j] = m.vec[i][j];
        }
    }
    return ans;
}

// get a Matrix* with the vec corresponding to the double* arr
Matrix* arr_to_mat(double* arr, int rows, int cols){
	Matrix* ans = new Matrix(rows, cols, "fixed", 0);
	for(int i=0; i<rows; i++){
		for(int j=0; j<cols; j++){
			ans->vec[i][j] = arr[i * cols + j];
		}
	}
	return ans;
}

// deallocate memory and clear the vector<Matrix*>
void dealloClear(vector<Matrix*> &vec){
    for(int i=0; i<vec.size(); i++){
        delete vec[i];
    }
    vec.clear();
}

// Constructor for the class ffnn. Init W, B with random values, and
// other values as required.
ffnn::ffnn(int num_input, int num_output, int num_layers, int num_neurons){

    // Set the corresponding variables.
    this->num_input = num_input;
    this->num_output = num_output;
    this->num_layers = num_layers;
    this->num_neurons = num_neurons;

    // Initializing W with random values, and B to all 0's.
    Matrix* W1 = new Matrix(num_neurons, num_input, "rand", num_input + num_neurons);
    this->W.pb({});
    this->B.pb({});
    this->W.pb(W1);
    Matrix* B1 = new Matrix(num_neurons, 1, "fixed", 0);
    this->B.pb(B1);
    for(int i = 2; i < num_layers; i++){
        Matrix* Wi = new Matrix(num_neurons, num_neurons, "rand", num_neurons + num_neurons);
        Matrix* Bi = new Matrix(num_neurons, 1, "fixed", 0);
        this->W.pb(Wi);
        this->B.pb(Bi);
    }
    Matrix* Wl = new Matrix(num_output, num_neurons, "rand", num_output + num_neurons);
    Matrix* Bl = new Matrix(num_output, 1, "fixed", 0);
    this->W.pb(Wl);
    this->B.pb(Bl);
}

// Apply the activation function on the Matrix.
Matrix* ffnn::g(Matrix* Ai){

    // If Sigmoid
    if(activation_func == "sigmoid"){
        Matrix* hi = new Matrix(this->num_neurons, 1, "fixed", 0);
        for(int i = 0; i < this->num_neurons; i++){ 
            hi->vec[i][0] = 1.0 / (1.0 + exp(-Ai->vec[i][0]));
        }
        return hi;
    }

    // If tanh
    else if(activation_func == "tanh"){
        Matrix* hi = new Matrix(this->num_neurons, 1, "fixed", 0);
        for(int i = 0; i < this->num_neurons; i++){ 
            hi->vec[i][0] = tanh(Ai->vec[i][0]);
        }
        return hi;
    }

    // If ReLU
    else if(activation_func == "relu"){
        Matrix* hi = new Matrix(this->num_neurons, 1, "fixed", 0);
        for(int i = 0; i < this->num_neurons; i++){ 
            hi->vec[i][0] = max(Ai->vec[i][0], 0.0);
        }
        return hi;
    }
    return NULL;
}

// The softmax function
Matrix* ffnn::o(Matrix* Al){
    Matrix* out = new Matrix(this->num_output, 1, "fixed", 0);
    double s = 0;
    for(int i = 0; i < this->num_output; i++){
        s += exp(Al->vec[i][0]);
    }
    for(int i = 0; i < this->num_output; i++){
        out->vec[i][0] = 1.0 * exp(Al->vec[i][0]) / s;
    }
    return out;
}

// Returns a one hot 1-D Matrix.
Matrix* ffnn::e(int index){
    Matrix* ans = new Matrix(this->num_output, 1, "fixed", 0);
    ans->vec[index][0] = 1;
    return ans;
}

// Derivative of the activation function.
Matrix* ffnn::g1(Matrix* Ai){

    // If Sigmoid.
    if(activation_func == "sigmoid"){
        Matrix* hi = new Matrix(this->num_neurons, 1, "fixed", 0);
        for(int i = 0; i < this->num_neurons; i++){
            double s = 1.0 / (1.0 + exp(-Ai->vec[i][0]));
            hi->vec[i][0] = 1.0 * s * (1 - s);
        }
        return hi;
    }

    // If tanh
    else if(activation_func == "tanh"){
        Matrix* hi = new Matrix(this->num_neurons, 1, "fixed", 0);
        for(int i = 0; i < this->num_neurons; i++){ 
            hi->vec[i][0] = 1.0 - tanh(Ai->vec[i][0]) * tanh(Ai->vec[i][0]);
        }
        return hi;
    }

    // If ReLU
    else if(activation_func == "relu"){
        Matrix* hi = new Matrix(this->num_neurons, 1, "fixed", 0);
        for(int i = 0; i < this->num_neurons; i++){ 
            hi->vec[i][0] = (Ai->vec[i][0] > 0) ? 1 : 0;
        }
        return hi;
    }
    return NULL;
}

// Code for performing the forward propogation. Returns the final probability distribution 
// for each of the output number of labels.
Matrix* ffnn::forwardprop(int index){

    dealloClear(this->A);
    dealloClear(this->H);

    // Get the input data x_i's.
    Matrix* temp1 = arr_to_mat(x_train + index * 784, 1, 784);
    this->H.pb(temp1->reshape(784, 1));
    delete temp1;
    this->A.pb({});

    for(int i = 1; i < this->num_layers; i++){  
        Matrix* temp = (this->W[i]->matmul(this->H[i-1], this->usegpu));
        this->A.pb(temp->add(this->B[i], this->usegpu));
        delete temp;
        this->H.pb(g(this->A[i]));
    }
    Matrix *temp = (this->W[this->num_layers]->matmul(this->H[this->num_layers-1], this->usegpu));
    this->A.pb(temp->add(this->B[this->num_layers], this->usegpu));
    delete temp;

    // Get the final probability distributions
    Matrix* x = o(this->A[this->num_layers]);
    return x;
}

// Code for running the backpropogation algorithm. Updates the values of W, and B.
// Returns void.
void ffnn::backprop(Matrix* output, int index){
    dealloClear(this->da);
    dealloClear(this->dh);
    for(int i = 0; i < this->num_layers + 1; i++){
        this->da.pb({});
        this->dh.pb({});
    }
    Matrix* dal;
    if(this->loss_func == "cross_entropy"){
        Matrix* temp = e(y_train[index]);
        dal = output->subtract(temp, this->usegpu);
        delete temp;
    }

    this->da[this->num_layers] = dal;
    for(int i = this->num_layers; i > 0; i--){

        // If no value already set, init.
        if(this->dw[i]->vec[0][0] == -100){             
            Matrix* temp = this->H[i-1]->T(this->usegpu);
            this->dw[i] = this->da[i]->matmul(temp, this->usegpu);
            delete temp;
            this->db[i] = this->da[i]->cmult(1, this->usegpu);
        }

        // Else add up the gradients.
        else{
            Matrix* temp1 = this->H[i-1]->T(this->usegpu);
            Matrix* temp2 = this->da[i]->matmul(temp1, this->usegpu);
            Matrix *ndw = this->dw[i]->add(temp2, this->usegpu);
            delete this->dw[i];
            delete temp1;
            delete temp2;
            this->dw[i] = ndw;
            Matrix* ndb = this->db[i]->add(this->da[i], this->usegpu);
            delete this->db[i];
            this->db[i] = ndb;
        }
        Matrix* temp1 = (this->W[i]->T(this->usegpu));
        this->dh[i-1] = temp1->matmul(this->da[i], this->usegpu);
        delete temp1;
        if(i!=1){
            Matrix* temp1 = g1(this->A[i-1]);
            Matrix* temp2 = temp1->reshape(this->num_neurons, 1);
            this->da[i-1] = this->dh[i-1]->multiply(temp2, this->usegpu);
            delete temp1;
            delete temp2;
        }
    }
}

// Returns the loss.
double ffnn::loss(Matrix* output, int index){
    if(this->loss_func == "cross_entropy"){
        return -log(output->vec[y_train[index]][0]);
    }
    return 0;
}

// Runs the current configuration of W's and B's on the validation and
// the test data, and returns the corresponding accuracy.
void ffnn::accuracy(string s){
    double acc = 0;

    for(int index = 0; index < 10000; index++){
        dealloClear(this->A);
        dealloClear(this->H);
        Matrix* temp;
        
        // Get the data.
        if(s == "Validation")temp = arr_to_mat(x_val + index * 784, 1, 784);
        else if(s == "Test")temp = arr_to_mat(x_test + index * 784, 1, 784);
        this->H.pb(temp->reshape(784, 1));
        delete temp;
        this->A.pb({});

        // Forward Prop
        for(int i = 1; i < this->num_layers; i++){
            Matrix *temp1 = (this->W[i]->matmul(this->H[i-1], this->usegpu));
            this->A.pb(temp1->add(this->B[i], this->usegpu));
            delete temp1;
            H.pb(g(A[i]));
        }
        Matrix *temp1 = (this->W[this->num_layers]->matmul(this->H[this->num_layers-1], this->usegpu));
        this->A.pb(temp1->add(this->B[this->num_layers], this->usegpu));
        delete temp1;

        // Probability distribution
        Matrix* output = o(A[this->num_layers]);

        // Get the index corresponding to max probability.
        int ind = 0;
        double val = 0;
        for(int i = 0; i < this->num_output; i++){
            if(val < output->vec[i][0]){
                ind = i;
                val = output->vec[i][0];
            }
        }
        delete output;

        // Update accuracy
        if(s == "Validation")acc += (ind == y_val[index]) ? 1 : 0;
        else if(s == "Test")acc += (ind == y_test[index]) ? 1 : 0;
    }
    cout << s << " accuracy : " << 1.0 * acc/100 << " %" << endl;   
}

// Train function. Does forward prop, then backward prop, updates W, B, and gets the accuracy.
// Using mini-batch method.
void ffnn::train(string optimizer, int num_training, int num_epochs, int batch_size, double learning_rate){

    cout << "Training Started, Epochs : " << num_epochs << endl;

    if(optimizer == "sgd"){

        for(int inp = 0; inp < num_epochs; inp++){

            cout << "Epoch Number : " << inp + 1 << " - ";
            double total_loss = 0;
            dealloClear(this->dw);
            dealloClear(this->db);
            for(int i = 0; i < this->num_layers + 1; i++){
                Matrix* fw = new Matrix(1, 1, "fixed", -100);
                this->dw.pb(fw);
                Matrix* fb = new Matrix(1,1,"fixed",-100);
                this->db.pb(fb);
            }

            for(int i = 0; i < num_training; i++){
                Matrix* output = this->forwardprop(i);
                double curr_loss = this->loss(output, i);
                total_loss += curr_loss;
                this->backprop(output, i);
                delete output;

                // Updating Weights and Biases according to sgd update rule                
                if(i > 0 && i % batch_size == 0){

                    // update W according to the update rule.
                    // W -= dw * learning_rate
                    for(int j = 1; j < this->num_layers + 1; j++){
                        Matrix* temp1 = this->dw[j]->cmult(learning_rate, this->usegpu);
                        Matrix* newW = this->W[j]->subtract(temp1, this->usegpu);
                        delete this->W[j];
                        delete temp1;
                        this->W[j] = newW;
                    }

                    // update B according to the update rule.
                    // B -= db * learning_rate
                    for(int j = 1; j < this->num_layers + 1; j++){
                        Matrix* temp = this->db[j]->cmult(learning_rate, this->usegpu);
                        Matrix* newB = this->B[j]->subtract(temp, this->usegpu);
                        delete this->B[j];
                        delete temp;
                        this->B[j] = newB;
                    }
                    dealloClear(this->dw);
                    dealloClear(this->db);
                    for(int i = 0; i < this->num_layers + 1; i++){
                        Matrix* fw = new Matrix(1, 1, "fixed", -100);
                        this->dw.pb(fw);
                        Matrix* fb = new Matrix(1,1,"fixed",-100);
                        this->db.pb(fb);
                    }
                }
            }

            this->accuracy("Validation");
        }
    }

    else if(optimizer == "mbgd"){

    	vector<Matrix*> prev_w, prev_b;
        prev_w.pb({});
        prev_b.pb({});
        for(int i = 1; i < this->num_layers + 1; i++){
            Matrix* fw = new Matrix(this->W[i]->rows, this->W[i]->cols, "fixed", 0);
            prev_w.pb(fw);
            Matrix* fb = new Matrix(this->B[i]->rows, this->B[i]->cols, "fixed", 0);
            prev_b.pb(fb);
        }

        // Using recommended values of hyperparameters
    	double gamma = 0.9;
    	for(int inp = 0; inp < num_epochs; inp++){
            cout << "Epoch Number : " << inp + 1 << " - ";
    		double total_loss = 0;
    		dealloClear(this->dw);
            dealloClear(this->db);
    		for(int i = 0; i < this->num_layers + 1; i++){
                Matrix* fw = new Matrix(1, 1, "fixed", -100);
                this->dw.pb(fw);
                Matrix* fb = new Matrix(1,1,"fixed",-100);
                this->db.pb(fb);
            }

            for(int i = 0; i < num_training; i++){
                Matrix* output = this->forwardprop(i);
                double curr_loss = this->loss(output, i);
                total_loss += curr_loss;
                this->backprop(output, i);
                delete output;

                // Updating Weights and Biases according to mbgd update rule
                if(i > 0 && i % batch_size == 0){

                    // update W according to the update rule:
                    // v_w = prev_w * val + dw * learning_rate
                    // W -= prev_w * val
                    // prev_w = v_w
                    for(int j = 1; j < this->num_layers + 1; j++){
                        double val = gamma ;
                        Matrix *temp1 = prev_w[j]->cmult(val,this->usegpu); 
                        Matrix *temp2 = this->dw[j]->cmult(learning_rate, this->usegpu);
                        Matrix *v_w = temp1->add(temp2, this->usegpu);
                        delete temp1; delete temp2; 
                        temp1 = this->W[j]->subtract(v_w, this->usegpu);
                        delete prev_w[j]; 
                        delete this->W[j];
                        this->W[j] = temp1;
                        prev_w[j] = v_w;
                    }
                    
                    // update B according to the update rule:
                    // v_b = prev_b * val + db * learning_rate
                    // B -= prev_b * val
                    // prev_b = v_b
                    for(int j = 1; j < this->num_layers + 1; j++){
                        double val = gamma ;
                        Matrix *temp1 =prev_b[j]->cmult(val,this->usegpu); 
                        Matrix *temp2 = this->db[j]->cmult(learning_rate, this->usegpu);
                        Matrix *v_b = temp1->add(temp2, this->usegpu);
                        delete temp1; delete temp2; 
                        temp1 = this->B[j]->subtract(v_b, this->usegpu);
                        delete prev_b[j]; 
                        delete this->B[j];
                        this->B[j] = temp1;
                        prev_b[j] = v_b;
                    }

                    dealloClear(this->dw);
                    dealloClear(this->db);
                    for(int j = 0; j < this->num_layers + 1; j++){
                        Matrix* fw = new Matrix(1, 1, "fixed", -100);
                        this->dw.pb(fw);
                        Matrix* fb = new Matrix(1,1,"fixed",-100);
                        this->db.pb(fb);
                    }
                } 
            }

            this->accuracy("Validation");
    	}
    }

    else if(optimizer == "rmsprop"){

    	// using recommended values of hyperparameters
        double beta = 0.1, eps = 1e-8, beta1 = 0.9;
        vector<Matrix*> v_w, v_b;

        for(int inp = 0; inp < num_epochs; inp++){
            cout << "Epoch Number : " << inp + 1 << " - ";
            double total_loss = 0;
            dealloClear(this->dw);
            dealloClear(this->db);
            for(int i = 0; i < this->num_layers + 1; i++){
                Matrix* fw = new Matrix(1, 1, "fixed", -100);
                this->dw.pb(fw);
                Matrix* fb = new Matrix(1, 1, "fixed", -100);
                this->db.pb(fb);
            }

            for(int i = 0; i < num_training; i++){
                Matrix* output = this->forwardprop(i);
                double curr_loss = this->loss(output, i);
                total_loss += curr_loss;
                this->backprop(output, i);
                delete output;

                // updating Weights and Biases according to rmsprop update rule
                if(i > 0 && i % batch_size == 0){

                	// intitialize v_w and v_b
                    if(inp==0){
                        for(int j = 0; j < this->num_layers + 1; j++){
                            Matrix* fw = new Matrix(dw[j]->rows, dw[j]->cols, "fixed", 0);
                            v_w.pb(fw);
                            Matrix* fb = new Matrix(db[j]->rows, db[j]->cols, "fixed", 0);
                            v_b.pb(fb);
                        }
                    }

                    // update W according to the update rule:
                    //  v_w = beta1 * v_w + (1 - beta) * (dw)^2
                    //  W -= (eta/sqrt(eps + v_w)) * dw 
                    for(int j = 1; j < this->num_layers + 1; j++){
                        Matrix* temp1 = v_w[j]->cmult(beta1, this->usegpu);
                        Matrix* temp2 = dw[j]->square(this->usegpu);
                        Matrix* temp3 = temp2->cmult((1.0-beta), this->usegpu);
                        delete v_w[j];
                        v_w[j] = temp1->add(temp3, this->usegpu);
                        delete temp1; delete temp2; delete temp3;
                        Matrix* temp4 = v_w[j]->cadd(eps, this->usegpu);
                        Matrix* temp5 = temp4->msqrt(this->usegpu);
                        Matrix* temp6 = temp5->elinv(this->usegpu);
                        delete temp4; delete temp5;
                        Matrix* temp7 = temp6->cmult(learning_rate, this->usegpu);
                        Matrix* temp8 = temp7->multiply(dw[j], this->usegpu);
                        delete temp7; delete temp6;
                        Matrix* newW = this->W[j]->subtract(temp8, this->usegpu);
                        delete this->W[j];
                        this->W[j] = newW;
                        delete temp8;
                    }

                    // update B according to the update rule:
                    //  v_b = beta1 * v_b + (1 - beta) * (db)^2
                    //  B -= (eta/sqrt(eps + v_b)) * db 
                    for(int j = 1; j < this->num_layers + 1; j++){
                        Matrix* temp1 = v_b[j]->cmult(beta1, this->usegpu);
                        Matrix* temp2 = db[j]->square(this->usegpu);
                        Matrix* temp3 = temp2->cmult((1.0-beta), this->usegpu);
                        delete v_b[j];
                        v_b[j] = temp1->add(temp3, this->usegpu);
                        delete temp1; delete temp2; delete temp3;
                        Matrix* temp4 = v_b[j]->cadd(eps, this->usegpu);
                        Matrix* temp5 = temp4->msqrt(this->usegpu);
                        Matrix* temp6 = temp5->elinv(this->usegpu);
                        delete temp4; delete temp5;
                        Matrix* temp7 = temp6->cmult(learning_rate, this->usegpu);
                        Matrix* temp8 = temp7->multiply(db[j], this->usegpu);
                        delete temp7; delete temp6;
                        Matrix* newB = this->B[j]->subtract(temp8, this->usegpu);
                        delete this->B[j];
                        this->B[j] = newB;
                        delete temp8;
                    }

                    // resetting dw and db
                    dealloClear(this->dw);
                    dealloClear(this->db);
                    for(int i = 0; i < this->num_layers + 1; i++){
                        Matrix* fw = new Matrix(1, 1, "fixed", -100);
                        this->dw.pb(fw);
                        Matrix* fb = new Matrix(1,1,"fixed",-100);
                        this->db.pb(fb);
                    }
                }
            }
            this->accuracy("Validation");
        }
    }
    else if(optimizer == "adam"){

    	// using recommended values of hyperparameters
        double beta1 = 0.9, eps = 1e-8, beta2 = 0.999;
        vector<Matrix*> v_w, v_b, m_w, m_b;
        for(int inp = 0; inp < num_epochs; inp++){
            cout << "Epoch Number : " << inp + 1 << " - ";
            double total_loss = 0;
            dealloClear(this->dw);
            dealloClear(this->db);
            for(int i = 0; i < this->num_layers + 1; i++){
                Matrix* fw = new Matrix(1, 1, "fixed", -100);
                this->dw.pb(fw);
                Matrix* fb = new Matrix(1, 1, "fixed", -100);
                this->db.pb(fb);
            }
            for(int i = 0; i < num_training; i++){
                Matrix* output = this->forwardprop(i);
                double curr_loss = this->loss(output, i);
                total_loss += curr_loss;
                this->backprop(output, i);
                delete output;

                // updating weights and biases according to adam update rules
                if(i > 0 && i % batch_size == 0){
                    
                	// initialising v_w, v_b and m_w, m_b
                    if(inp==0){
                        for(int j = 0; j < this->num_layers + 1; j++){
                            Matrix* fw = new Matrix(dw[j]->rows, dw[j]->cols, "fixed", 0);
                            Matrix* fw1 = new Matrix(dw[j]->rows, dw[j]->cols, "fixed", 0);
                            v_w.pb(fw);
                            m_w.pb(fw1);
                            Matrix* fb = new Matrix(db[j]->rows, db[j]->cols, "fixed", 0);
                            Matrix* fb1 = new Matrix(db[j]->rows, db[j]->cols, "fixed", 0);
                            v_b.pb(fb);
                            m_b.pb(fb1);
                        }
                    }

                    // updating W according to the update rule:
                    // m_w = beta1 * m_w + (1-beta1) * dw
        			// v_w = beta2 * v_w + (1-beta2) * (dw)^2
        			// m_w_cap = m_w/(1-beta1**(t+1))
        			// v_w_cap = v_w/(1-beta2**(t+1))
        			// w = w - (eta/(sqrt(v_w_cap+eps)))*(m_w_cap)
                    for(int j = 1; j < this->num_layers + 1; j++){
                        Matrix* temp1 = v_w[j]->cmult(beta2, this->usegpu);
                        Matrix* temp2 = dw[j]->square(this->usegpu);
                        Matrix* temp3 = temp2->cmult((1.0-beta2), this->usegpu);
                        delete v_w[j];
                        v_w[j] = temp1->add(temp3, this->usegpu);
                        delete temp1; delete temp2; delete temp3;
                        temp1 = m_w[j]->cmult(beta1, this->usegpu);
                        temp2 = dw[j]->cmult((1-beta1), this->usegpu);
                        delete m_w[j];
                        m_w[j] = temp1->add(temp2, this->usegpu);
                        delete temp1; delete temp2;
                        double val = (1 - pow(beta1, (i/batch_size)+1));
                        Matrix* m_w_cap = m_w[j]->cmult(1.0/val, this->usegpu);
                        val = (1 - pow(beta2, (i/batch_size)+1));
                        Matrix* v_w_cap = v_w[j]->cmult(1.0/val, this->usegpu);
                        Matrix* temp4 = v_w_cap->cadd(eps, this->usegpu);
                        Matrix* temp5 = temp4->msqrt(this->usegpu);
                        Matrix* temp6 = temp5->elinv(this->usegpu);
                        delete v_w_cap; delete temp4; delete temp5;
                        Matrix* temp7 = temp6->cmult(learning_rate, this->usegpu);
                        Matrix* temp8 = temp7->multiply(m_w_cap, this->usegpu);
                        delete temp7; delete temp6;
                        Matrix *newW = this->W[j]->subtract(temp8, this->usegpu);
                        delete this->W[j];
                        this->W[j] = newW;
                        delete m_w_cap; delete temp8;
                    }

                    // updating B according to the update rule:
                    // m_b = beta1 * m_b + (1-beta1) * db
        			// v_b = beta2 * v_b + (1-beta2) * (db)^2
        			// m_b_cap = m_b/(1-beta1**(t+1))
        			// v_b_cap = v_b/(1-beta2**(t+1))
        			// w = w - (eta/(sqrt(v_b_cap + eps)))*(m_b_cap)
                    for(int j = 1; j < this->num_layers + 1; j++){
                        Matrix* temp1 = v_b[j]->cmult(beta2, this->usegpu);
                        Matrix* temp2 = db[j]->square(this->usegpu);
                        Matrix* temp3 = temp2->cmult((1.0-beta2), this->usegpu);
                        delete v_b[j];
                        v_b[j] = temp1->add(temp3, this->usegpu);
                        delete temp1; delete temp2; delete temp3;
                        temp1 = m_b[j]->cmult(beta1, this->usegpu);
                        temp2 = db[j]->cmult((1-beta1), this->usegpu);
                        delete m_b[j];
                        m_b[j] = temp1->add(temp2, this->usegpu);
                        delete temp1; delete temp2;
                        double val = (1 - pow(beta1, (i/batch_size)+1));
                        Matrix* m_b_cap = m_b[j]->cmult(1.0/val, this->usegpu);
                        val = (1 - pow(beta2, (i/batch_size)+1));
                        Matrix* v_b_cap = v_b[j]->cmult(1.0/val, this->usegpu);
                        Matrix* temp4 = v_b_cap->cadd(eps, this->usegpu);
                        Matrix* temp5 = temp4->msqrt(this->usegpu);
                        Matrix* temp6 = temp5->elinv(this->usegpu);
                        delete v_b_cap; delete temp4; delete temp5;
                        Matrix* temp7 = temp6->cmult(learning_rate, this->usegpu);
                        Matrix* temp8 = temp7->multiply(m_b_cap, this->usegpu);
                        delete temp7; delete temp6;
                        Matrix* newB = this->B[j]->subtract(temp8, this->usegpu);
                        delete this->B[j];
                        this->B[j] = newB;
                        delete m_b_cap; delete temp8;
                    }

                    // resetting dw and db
                    dealloClear(this->dw);
                    dealloClear(this->db);
                    for(int i = 0; i < this->num_layers + 1; i++){
                        Matrix* fw = new Matrix(1, 1, "fixed", -100);
                        this->dw.pb(fw);
                        Matrix* fb = new Matrix(1,1,"fixed",-100);
                        this->db.pb(fb);
                    }
                }
            }
            this->accuracy("Validation");
        }
    }
    cout << "Training Done ..." << endl;
    cout << "Running on Test Data now ..." << endl;
    this->accuracy("Test");
}

// Return the time difference between the two timeval instances.
double time_diff(struct timeval t2, struct timeval t1){
    return (1000000.0*(t2.tv_sec-t1.tv_sec) + t2.tv_usec-t1.tv_usec)/1000000.0;
}

// Reshape given object to required shape
Matrix* Matrix::reshape(int rows, int cols){
    Matrix* ans = new Matrix(rows, cols, "fixed", 0);
    for(int i = 0; i < rows; i++){
        for(int j = 0; j < cols; j++){
            int index1 = (i * cols + j) / this->cols;
            int index2 = (i * cols + j) % this->cols;
            ans->vec[i][j] = this->vec[index1][index2];
        }
    }
    return ans;
}

// Add two matrices
Matrix* Matrix::add(Matrix* m2, bool usegpu= true){
    if(usegpu){
        gettimeofday(&t1, 0);
    	int num_blocks = 64;
    	int rows = this->rows, cols = this->cols;
    	int num_threads = ceil(1.0*rows*cols/num_blocks);
    	double *arr1, *arr2, *ans, *gpuans;
    	cudaMalloc(&gpuans, rows * cols * sizeof(double));
        double *temp1 = vec_to_arr(*this);
    	arr1 = get_gpu(temp1, rows * cols);
        double *temp2 = vec_to_arr(*m2);
    	arr2 = get_gpu(temp2, m2->rows * m2->cols);
        delete temp1; delete temp2;
        gettimeofday(&t2, 0);
    	gpu_add <<< num_threads, num_blocks >>> (arr1, arr2, gpuans, rows * cols);
    	cudaDeviceSynchronize();
    	gettimeofday(&t3, 0);
        ans = get_cpu<double>(gpuans, rows * cols);
        cudaFree(gpuans); cudaFree(arr1); cudaFree(arr2);
    	Matrix *a = arr_to_mat(ans, rows, cols);
        delete ans;
        gettimeofday(&t4, 0);
        gpu_function_time += time_diff(t4, t1);
        gpu_kernel_time += time_diff(t3, t2);
        return a;
	}
	else{
        gettimeofday(&t1, 0);
		int rows = this->rows, cols=this->cols;
		Matrix* m3 = new Matrix(rows, cols, "fixed", 0);
		for(int i=0; i<rows; i++){
			for(int j=0; j<cols; j++){
				m3->vec[i][j] = this->vec[i][j] + m2->vec[i][j];
			}
		}
        gettimeofday(&t2, 0);
        cpu_function_time += time_diff(t2, t1);
		return m3;
	}
}

// Subtract two matrices
Matrix* Matrix::subtract(Matrix* m2, bool usegpu= true){
    if(usegpu){
        gettimeofday(&t1, 0);
    	int num_blocks = 64;
    	int rows = this->rows, cols = this->cols;
    	int num_threads = ceil(1.0*rows*cols/num_blocks);
    	double* arr1, *arr2, *ans, *gpuans;
    	cudaMalloc(&gpuans, rows * cols * sizeof(double));
    	double *temp1 = vec_to_arr(*this);
        arr1 = get_gpu(temp1, rows * cols);
        double *temp2 = vec_to_arr(*m2);
        arr2 = get_gpu(temp2, m2->rows * m2->cols);
        delete temp1; delete temp2;
        gettimeofday(&t2, 0);
    	gpu_subtract <<< num_threads, num_blocks >>> (arr1, arr2, gpuans, rows * cols);
    	cudaDeviceSynchronize();
        gettimeofday(&t3, 0);
    	ans = get_cpu<double>(gpuans, rows * cols);
        cudaFree(gpuans); cudaFree(arr1); cudaFree(arr2);
    	Matrix* a = arr_to_mat(ans, rows, cols);
        delete ans;
        gettimeofday(&t4, 0);
        gpu_function_time += time_diff(t4, t1);
        gpu_kernel_time += time_diff(t3, t2);
        return a;
	}
	else{
        gettimeofday(&t1, 0);
		int rows = this->rows, cols=this->cols;
		Matrix* m3 = new Matrix(rows, cols, "fixed", 0);
		for(int i=0; i<rows; i++){
			for(int j=0; j<cols; j++){
				m3->vec[i][j] = this->vec[i][j] - m2->vec[i][j];
			}
		}
        gettimeofday(&t2, 0);
        cpu_function_time += time_diff(t2, t1);
		return m3;
	}
}

// Hadamard product of two matrices
Matrix* Matrix::multiply(Matrix* m2, bool usegpu=true){
    if(usegpu){
        gettimeofday(&t1, 0);
    	int num_blocks = 64;
    	int rows = this->rows, cols = this->cols;
    	int num_threads = ceil(1.0*rows*cols/num_blocks);
    	double* arr1, *arr2, *ans, *gpuans;
    	cudaMalloc(&gpuans, rows * cols * sizeof(double));
    	double *temp1 = vec_to_arr(*this);
        arr1 = get_gpu(temp1, rows * cols);
        double *temp2 = vec_to_arr(*m2);
        arr2 = get_gpu(temp2, m2->rows * m2->cols);
        delete temp1; delete temp2;
        gettimeofday(&t2, 0);
    	gpu_mult <<< num_threads, num_blocks >>> (arr1, arr2, gpuans, rows * cols);
        cudaDeviceSynchronize();
        gettimeofday(&t3, 0);
    	ans = get_cpu<double>(gpuans, rows * cols);
	    cudaFree(arr1); cudaFree(arr2); cudaFree(gpuans);
	    Matrix* a = arr_to_mat(ans, rows, cols);
        delete ans;
        gettimeofday(&t4, 0);
        gpu_function_time += time_diff(t4, t1);
        gpu_kernel_time += time_diff(t3, t2);
        return a;
	}
	else{
        gettimeofday(&t1, 0);
		int rows = this->rows, cols=this->cols;
		Matrix* m3 = new Matrix(rows, cols, "fixed", 0);
		for(int i=0; i<rows; i++){
			for(int j=0; j<cols; j++){
				m3->vec[i][j] = this->vec[i][j] * m2->vec[i][j];
			}
		}
        gettimeofday(&t2, 0);
        cpu_function_time += time_diff(t2, t1);
		return m3;
	}
}

// Inner product of 2 vectors, DEPRECATED
double Matrix::vecdot(Matrix* m1, Matrix* m2, bool usegpu=true){
    //cout<<"vecdot"<<endl;
    int num_blocks = 64;
    int num_threads = ceil(m1->rows * m1->cols/num_blocks);
    double* arr1, *arr2, *ans, *gpuans;
    cudaMalloc(&gpuans, sizeof(double));
    arr1 = get_gpu(vec_to_arr(*m1), m1->rows * m1->cols);
    arr2 = get_gpu(vec_to_arr(*m2), m2->rows * m2->cols);
    gpu_vecdot <<< num_blocks, num_threads >>> (arr1, arr2, gpuans, m1->rows * m1->cols);
    cudaDeviceSynchronize();
    ans = get_cpu<double>(gpuans, 1);

    return ans[0];
}

// Matrix multiplication
Matrix* Matrix::matmul(Matrix* m2, bool usegpu=true){
	if(usegpu){
        gettimeofday(&t1, 0);
		int num_blocks = 8;
		int rows = this->rows, cols = this->cols;
		int num_threads = ceil(1.0 * rows * m2->cols / num_blocks);
		double* arr1, *arr2, *ans, *gpuans;
		// unsigned int grid_rows = (rows+num_blocks-1)/num_blocks;
		// unsigned int grid_cols = (m2->cols+num_blocks-1)/num_blocks;
		dim3 dimGrid(100, 100);
		dim3 dimBlock(num_blocks, num_blocks);
    	cudaMalloc(&gpuans, rows * m2->cols * sizeof(double));
    	double *temp1 = vec_to_arr(*this);
        arr1 = get_gpu(temp1, rows * cols);
        double *temp2 = vec_to_arr(*m2);
        arr2 = get_gpu(temp2, m2->rows * m2->cols);
        delete temp1; delete temp2;
        gettimeofday(&t2, 0);
    	gpu_matmul <<< dimGrid, dimBlock>>> (arr1, arr2, gpuans, rows, cols, m2->cols);
    	cudaDeviceSynchronize();
        gettimeofday(&t3, 0);
    	ans = get_cpu<double>(gpuans, rows * m2->cols);
	    cudaFree(arr1); cudaFree(arr2); cudaFree(gpuans);
	    Matrix* a = arr_to_mat(ans, rows, m2->cols);
        delete ans;
        gettimeofday(&t4, 0);
        gpu_function_time += time_diff(t4, t1);
        gpu_kernel_time += time_diff(t3, t2);
        return a;
	}
	else{
        gettimeofday(&t1, 0);
		Matrix* m3 = new Matrix(this->rows, m2->cols, "fixed", 0);
    	for(int i=0; i<this->rows; i++){
	        for(int j=0; j<m2->cols; j++){
            	for(int k=0; k<this->cols; k++){
	                m3->vec[i][j] += this->vec[i][k] * m2->vec[k][j];
            	}
            }
    	}
        gettimeofday(&t2, 0);
        cpu_function_time += time_diff(t2, t1);
    	return m3;
	}
}

// Transpose of a matrix
Matrix* Matrix::T(bool usegpu=true){
	if(usegpu){
        gettimeofday(&t1, 0);
		int num_blocks = 64;
        int rows = this->rows, cols = this->cols;
		int num_threads = ceil(1.0*(this->rows)*(this->cols)/num_blocks);
		double* arr1, *ans, *gpuans;
		cudaMalloc(&gpuans, (this->rows)*(this->cols)*sizeof(double));
		double *temp1 = vec_to_arr(*this);
        arr1 = get_gpu(temp1, rows * cols);
        delete temp1;
        gettimeofday(&t2, 0);
		gpu_T <<< num_threads, num_blocks >>> (arr1, gpuans, this->rows, this->cols);
		cudaDeviceSynchronize();
        gettimeofday(&t3, 0);
		ans = get_cpu<double>(gpuans, this->rows*this->cols);
        cudaFree(arr1); cudaFree(gpuans);
		Matrix* a = arr_to_mat(ans, cols, rows);
        delete ans;
        gettimeofday(&t4, 0);
        gpu_function_time += time_diff(t4, t1);
        gpu_kernel_time += time_diff(t3, t2);
        return a;
	}
	else{
        gettimeofday(&t1, 0);
		Matrix* m2 = new Matrix(this->cols, this->rows, "fixed", 0);
		for(int i=0; i<this->rows; i++){
			for(int j=0; j<this->cols; j++){
				m2->vec[j][i] = this->vec[i][j];
			}
		}
        gettimeofday(&t2, 0);
        cpu_function_time += time_diff(t2, t1);
		return m2;
	}
}

// Squaring all elements in Matrix
Matrix* Matrix::square(bool usegpu=true){
    if(usegpu){
        gettimeofday(&t1, 0);
        int num_blocks = 64;
        int rows = this->rows, cols = this->cols;
        int num_threads = ceil(1.0*(this->rows)*(this->cols)/num_blocks);
        double* arr1, *ans, *gpuans;
        cudaMalloc(&gpuans, (this->rows)*(this->cols)*sizeof(double));
        double *temp1 = vec_to_arr(*this);
        arr1 = get_gpu(temp1, rows * cols);
        delete temp1;
        gettimeofday(&t2, 0);
        gpu_square <<< num_threads, num_blocks >>> (arr1, gpuans, this->rows*this->cols);
        cudaDeviceSynchronize();
        gettimeofday(&t3, 0);
        ans = get_cpu<double>(gpuans, this->rows*this->cols);
        cudaFree(arr1); cudaFree(gpuans);
        Matrix* a = arr_to_mat(ans, rows, cols);
        delete ans;
        gettimeofday(&t4, 0);
        gpu_function_time += time_diff(t4, t1);
        gpu_kernel_time += time_diff(t3, t2);
        return a;
    }
    else{
        gettimeofday(&t1, 0);
        Matrix* m2 = new Matrix(this->rows, this->cols, "fixed", 0);
        for(int i=0; i<this->rows; i++){
            for(int j=0; j<this->cols; j++){
                m2->vec[i][j] = (this->vec[i][j])*(this->vec[i][j]);
            }
        }
        gettimeofday(&t2, 0);
        cpu_function_time += time_diff(t2, t1);
        return m2;
    }
}

// Sqrt all elements in Matrix
Matrix* Matrix::msqrt(bool usegpu=true){
    if(usegpu){
        gettimeofday(&t1, 0);
        int num_blocks = 64;
        int rows = this->rows, cols = this->cols;
        int num_threads = ceil(1.0*(this->rows)*(this->cols)/num_blocks);
        double* arr1, *ans, *gpuans;
        cudaMalloc(&gpuans, (this->rows)*(this->cols)*sizeof(double));
        double *temp1 = vec_to_arr(*this);
        arr1 = get_gpu(temp1, rows * cols);
        delete temp1;
        gettimeofday(&t2, 0);
        gpu_sqrt <<< num_threads, num_blocks >>> (arr1, gpuans, this->rows*this->cols);
        cudaDeviceSynchronize();
        gettimeofday(&t3, 0);
        ans = get_cpu<double>(gpuans, this->rows*this->cols);
        cudaFree(arr1); cudaFree(gpuans);
        Matrix* a = arr_to_mat(ans, rows, cols);
        delete ans;
        gettimeofday(&t4, 0);
        gpu_function_time += time_diff(t4, t1);
        gpu_kernel_time += time_diff(t3, t2);
        return a;
    }
    else{
        gettimeofday(&t1, 0);
        Matrix* m2 = new Matrix(this->rows, this->cols, "fixed", 0);
        for(int i=0; i<this->rows; i++){
            for(int j=0; j<this->cols; j++){
                m2->vec[i][j] = sqrt(this->vec[i][j]);
            }
        }
        gettimeofday(&t2, 0);
        cpu_function_time += time_diff(t2, t1);
        return m2;
    }
}

// Inverse all elements in Matrix
Matrix* Matrix::elinv(bool usegpu=true){
    if(usegpu){
        gettimeofday(&t1, 0);
        int num_blocks = 64;
        int rows = this->rows, cols = this->cols;
        int num_threads = ceil(1.0*(this->rows)*(this->cols)/num_blocks);
        double* arr1, *ans, *gpuans;
        cudaMalloc(&gpuans, (this->rows)*(this->cols)*sizeof(double));
        double *temp1 = vec_to_arr(*this);
        arr1 = get_gpu(temp1, rows * cols);
        delete temp1;
        gettimeofday(&t2, 0);
        gpu_elinv <<< num_threads, num_blocks >>> (arr1, gpuans, this->rows*this->cols);
        cudaDeviceSynchronize();
        gettimeofday(&t3, 0);
        ans = get_cpu<double>(gpuans, this->rows*this->cols);
        cudaFree(arr1); cudaFree(gpuans);
        Matrix* a = arr_to_mat(ans, rows, cols);
        delete ans;
        gettimeofday(&t4, 0);
        gpu_function_time += time_diff(t4, t1);
        gpu_kernel_time += time_diff(t3, t2);
        return a;
    }
    else{
        gettimeofday(&t1, 0);
        Matrix* m2 = new Matrix(this->rows, this->cols, "fixed", 0);
        for(int i=0; i<this->rows; i++){
            for(int j=0; j<this->cols; j++){
                m2->vec[i][j] = 1.0/(this->vec[i][j]);
            }
        }
        gettimeofday(&t2, 0);
        cpu_function_time += time_diff(t2, t1);
        return m2;
    }
}

// Multiply all elements in matrix with a constant
Matrix* Matrix::cmult(double val, bool usegpu=true){
    if(usegpu){
        gettimeofday(&t1, 0);
        int num_blocks = 64;
        int rows = this->rows, cols = this->cols;
        double* arr1, *ans, *gpuans;
        int num_threads = ceil(1.0*rows*cols/num_blocks);
        cudaMalloc(&gpuans, rows * cols * sizeof(double));
        double *temp1 = vec_to_arr(*this);
        arr1 = get_gpu(temp1, rows * cols);
        delete temp1;
        gettimeofday(&t2, 0);
        gpu_cmult <<< num_threads, num_blocks >>> (arr1, gpuans, val, this->rows*this->cols);
        cudaDeviceSynchronize();
        gettimeofday(&t3, 0);
        ans = get_cpu<double>(gpuans, this->rows * this->cols);
        cudaFree(arr1); cudaFree(gpuans);
        Matrix* a = arr_to_mat(ans, rows, cols);
        delete ans;
        gettimeofday(&t4, 0);
        gpu_function_time += time_diff(t4, t1);
        gpu_kernel_time += time_diff(t3, t2);
        return a;
    }
    else{
        gettimeofday(&t1, 0);
        Matrix* m2 = new Matrix(this->rows, this->cols, "fixed", 0);
        for(int i=0; i<this->rows; i++){
            for(int j=0; j<this->cols; j++){
                m2->vec[i][j] = this->vec[i][j] * val;
            }
        }
        gettimeofday(&t2, 0);
        cpu_function_time += time_diff(t2, t1);
        return m2;
    }
}

// Add a constant to all elements of a Matrix
Matrix* Matrix::cadd(double val, bool usegpu=true){
    if(usegpu){
        gettimeofday(&t1, 0);
        int num_blocks = 64;
        double* arr1, *ans, *gpuans;
        int num_threads = ceil(1.0*rows*cols/num_blocks);
        cudaMalloc(&gpuans, rows * cols * sizeof(double));
        double *temp1 = vec_to_arr(*this);
        arr1 = get_gpu(temp1, rows * cols);
        delete temp1;
        gettimeofday(&t2, 0);
        gpu_cadd <<< num_threads, num_blocks >>> (arr1, gpuans, val, this->rows*this->cols);
        cudaDeviceSynchronize();
        gettimeofday(&t3, 0);
        ans = get_cpu<double>(gpuans, this->rows * this->cols);
        cudaFree(arr1); cudaFree(gpuans);
        Matrix* a = arr_to_mat(ans, rows, cols);
        delete ans;
        gettimeofday(&t4, 0);
        gpu_function_time += time_diff(t4, t1);
        gpu_kernel_time += time_diff(t3, t2);
        return a;
    }
    else{
        gettimeofday(&t1, 0);
        Matrix* m2 = new Matrix(this->rows, this->cols, "fixed", 0);
        for(int i=0; i<this->rows; i++){
            for(int j=0; j<this->cols; j++){
                m2->vec[i][j] = this->vec[i][j] + val;
            }
        }
        gettimeofday(&t2, 0);
        cpu_function_time += time_diff(t2, t1);
        return m2;
    }
}

// Prints the Matrix, for debugging
void Matrix::print(){

	for(int i=0; i<this->rows; i++){
		for(int j=0; j<this->cols; j++){
			cout<<this->vec[i][j]<<" ";
		}
		cout<<endl;
	}
}

// Prints shape of matrix, for debugging
void Matrix::shape(){
    cout<<this->rows<<" "<<this->cols<<endl;
}

// Compares two matrices, for debugging
bool Matrix::compare(Matrix *m2){
    for(int i=0; i<this->rows; i++){
        for(int j=0; j<this->cols; j++){
            if(abs(this->vec[i][j] - m2->vec[i][j]) > 1e-16){
                cout<<"Mismatch at ("<<i<<", "<<j<<"), error is "<<this->vec[i][j] - m2->vec[i][j]<<"\n";
                return false;
            }
        }
    }
    return true;
}

int main(int argc, char** argv){

    int input_layer_size = 784;
    int output_layer_size = 10;
    int num_hidden_layers = 2;
    int num_epochs = 3;
    int size_hlayer = 128;
    string activation_func = "relu";
    string loss_func = "cross_entropy";
    int batch_size = 32;
    double learning_rate = 0.001;
    string optimizer = "mbgd";
    string use = "both";
    int num_samples = 10000;

    // parsing command line arguments
    for(int i=1; i<argc-1; i+=2){
        if(argv[i][0]!='-'){
            cout<<"USAGE: ./a.out -u <use> -h <num_hidden_layers> -n <size_hlayer> -e <epochs> -s <num_samples> -l <learning_rate> -b <batch_size> -o <optimizer> -a <activation_func>\n";
            return 0;
        }
        switch(argv[i][1]){
            case 'h': num_hidden_layers = stoi(string(argv[i+1])); break;
            case 'n': size_hlayer = stoi(string(argv[i+1])); break;
            case 'e': num_epochs = stoi(string(argv[i+1])); break;
            case 's': num_samples = stoi(string(argv[i+1])); 
                    if(num_samples > 50000){
                        cout<<"Error: num_samples cannot be > 50000.\n";
                        return 0;
                    }
                    break;
            case 'l': learning_rate = stod(string(argv[i+1])); break;
            case 'b': batch_size = stoi(string(argv[i+1])); break;
            case 'o': optimizer = (string(argv[i+1])); 
                    if(optimizer!="sgd" && optimizer!="mbgd" && optimizer !="rmsprop" && optimizer!="adam"){
                        cout<<"Error: optimizer must be \'sgd\' or \'mbgd\' or \'rmsprop\' or \'adam\'.\n";
                        return 0;
                    }
                    break;
            case 'a': activation_func = (string(argv[i+1])); 
                    if(activation_func!="sigmoid" && activation_func!="tanh" && activation_func!="relu"){
                        cout<<"Error: activation_func must be \'sigmoid\' or \'tanh\' or \'relu\'.\n";
                        return 0;
                    }
                    break;
            case 'u': use = string(argv[i+1]); 
                    if(use!="cpu" && use!="gpu" && use!="both"){
                        cout<<"Error: use must be \'cpu\' or \'gpu\' or \'both\'.\n";
                        return 0;
                    }
                    break;
            default: cout<<"USAGE: ./a.out -u <use> -h <num_hidden_layers> -n <size_hlayer> -e <epochs> -s <num_samples> -l <learning_rate> -b <batch_size> -o <optimizer> -a <activation_func>\n";
                     return 0;
        }
    }
    cout << "Loading data ... " << endl;
    for(int i = 0; i < 50000; i++){
        fscanf(training_label, "%d", &y_train[i]);
        for(int j = 0; j < 784;j++){
            fscanf(training_img, "%lf", &x_train[i*784 + j]);
        }
    }
    for(int i = 0; i < 10000; i++){
        fscanf(test_label, "%d", &y_test[i]);
        for(int j = 0; j < 784;j++){
            fscanf(test_img, "%lf", &x_test[i*784 + j]);
        }
    }
    for(int i = 0; i < 10000; i++){
        fscanf(validation_label, "%d", &y_val[i]);
        for(int j = 0; j < 784;j++){
            fscanf(validation_img, "%lf", &x_val[i*784 + j]);
        }
    }
    cout << "Loading data done ..." << endl;

    if(use == "both" || use == "cpu"){
        cout << "\nUsing CPU" << endl;
        ffnn F(input_layer_size, output_layer_size, num_hidden_layers, size_hlayer);
        F.usegpu = false;
        F.activation_func = activation_func;
        F.loss_func = loss_func;
        cout << "CPU Model initialized" << endl;
        cout << "Using \n\tInput Layer Size : " << input_layer_size
                << "\n\tOutput Layer Size : " << output_layer_size
                << "\n\tNumber of Hidden Layers : " << num_hidden_layers
                << "\n\tSize of each Hidden layer : " << size_hlayer
                << "\n\tActivation function : " << activation_func
                << "\n\tOptimizer : " << optimizer
                << "\n\tNumber of training samples : "<<num_samples
                << "\n\tLoss Function : " << loss_func 
                << "\n\tBatch Size : " << batch_size
                << "\n\tLearning Rate : " << learning_rate << endl << endl;
        F.train(optimizer, num_samples, num_epochs, batch_size, learning_rate);
        cout << "CPU Execution time : " << cpu_function_time << " s " << endl;
    }
    if(use == "gpu" || use == "both"){ 
        cout << "\nUsing GPU" << endl;
        ffnn G(input_layer_size, output_layer_size, num_hidden_layers, size_hlayer);
        G.usegpu = true;
        G.activation_func = activation_func;
        G.loss_func = loss_func;
        cout << "GPU Model initialized" << endl;
        cout << "Using \n\tInput Layer Size : " << input_layer_size
                << "\n\tOutput Layer Size : " << output_layer_size
                << "\n\tNumber of Hidden Layers : " << num_hidden_layers
                << "\n\tSize of each Hidden layer : " << size_hlayer
                << "\n\tActivation function : " << activation_func
                << "\n\tOptimizer : " << optimizer
                << "\n\tNumber of training samples : "<<num_samples
                << "\n\tLoss Function : " << loss_func 
                << "\n\tBatch Size : " << batch_size
                << "\n\tLearning Rate : " << learning_rate << endl << endl;
        G.train(optimizer, num_samples, num_epochs, batch_size, learning_rate);
        cout << "GPU function execution time : " << gpu_function_time << " s " << endl;
        cout << "GPU kernel execution time : " << gpu_kernel_time << " s " << endl;
        cudaThreadSynchronize();
        return 0;
    }
}
